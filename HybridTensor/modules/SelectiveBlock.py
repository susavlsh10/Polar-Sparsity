from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None

class SelectBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls=None,
        mlp_cls=None,
        mlp_router=None,
        mha_router=None,
        norm_cls=nn.LayerNorm,
        dropout_cls=nn.Dropout,
        prenorm=True,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        drop_path1=0.0,
        drop_path2=0.0,
        fused_dropout_add_ln=False,
        return_residual=False,
        residual_in_fp32=False,
        sequence_parallel=False,
        mark_shared_params=False,
    ):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.

        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        Here we do: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, etc.

        If you want to do concurrency with CUDA graphs, your shapes must remain fixed 
        (batch_size, seq_len, etc.) across captures and replays. Also avoid any operations 
        that cause dynamic shape changes or memory allocations.
        """
        super().__init__()
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, "residual_in_fp32 is only compatible with prenorm=True"
        
        assert mixer_cls is not None and mlp_cls is not None, (
            "mixer_cls and mlp_cls cannot be None in SelectBlock"
        )
            
        # MHA & MLP submodules
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        self.total_neurons = self.mlp.fc1.weight.shape[0]
        
        # Routers
        if mlp_router is not None:
            self.mlp_router = mlp_router(dim)
            self.skip_attn_router = False
        else:
            self.mlp_router = None
            self.skip_attn_router = True
            
        if mha_router is not None:
            self.mha_router = mha_router(dim)
        else:
            self.mha_router = None

        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode="row")
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert layer_norm_fn is not None, "Triton layer_norm_fn not installed"
            assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(self.dropout1, nn.Dropout)

        # Mark the norm parameters for sequence parallel / shared params if needed
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, "norm2"):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, "norm2"):
                for p in self.norm2.parameters():
                    p._shared_params = True

        self.mlp_topk = None
        self.skip_mlp_router = False
        self.skip_attn_router = False
        
        # We'll use an extra stream for concurrency
        self.sparse_stream = torch.cuda.Stream(device="cuda", priority=0)
        self.main_stream = torch.cuda.Stream(device="cuda", priority=-5)
        # We'll record events to coordinate concurrency
        self.mha_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.mlp_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
        self.use_tensor_parallel = mark_shared_params
        
        if self.use_tensor_parallel:
            # save the stream and events in the mixer and mlp classes
            self.mlp.router = self.mlp_router
            self.mixer.router = self.mha_router
            
        self.mlp_topk_layers = None     # this will be a dictionary of layer_idx -> topk value
        self.attn_topk_layers = None    # this will be a dictionary of layer_idx -> topk value

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def prefill_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_kwargs=None, mixer_subset=None):
        hidden_states = self.mixer(hidden_states, **mixer_kwargs)
        
        if mixer_subset is not None:
            residual = residual[:, mixer_subset]

        if not isinstance(self.mlp, nn.Identity):
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path2(self.dropout2(hidden_states))
                if dropped.shape != residual.shape:
                    dropped = dropped.view(residual.shape)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                if hidden_states.shape != residual.shape:
                    hidden_states = hidden_states.view(residual.shape)
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    eps=self.norm2.eps,
                    dropout_p=self.dropout2.p if self.training else 0.0,
                    rowscale=rowscale2,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
    def decode_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_subset=None, mixer_kwargs=None):
        """ Single GPU Decode Forward

        Args:
            hidden_states (Tensor): _description_
            residual (Optional[Tensor], optional): _description_. Defaults to None.
            mixer_subset (_type_, optional): _description_. Defaults to None.
        """
        curr_stream = torch.cuda.current_stream()
        
        # We want to run MHA & mlp_router in parallel on different streams
        router_inputs = hidden_states.squeeze(1)  # shape (batch_size, dim)
        self.main_stream.wait_stream(curr_stream)
        self.sparse_stream.wait_stream(curr_stream)
        main_stream = self.main_stream
        
        # if mlp_topk > th * total_neurons, skip mlp router
        
        # if self.mlp_topk > 0.8 * self.total_neurons:
        #     self.skip_mlp_router = True
        # else:
        #     self.skip_mlp_router = False

        # [Sparse stream]  mlp_router
        if not self.skip_mlp_router:
            with torch.cuda.stream(self.sparse_stream):  
                index_vec = self.mlp_router._select_neurons_topk(router_inputs, topk = self.mlp_topk)
                self.sparse_stream.record_event(self.mlp_event)

        # [Main stream]  MHA
        with torch.cuda.stream(main_stream):
            batch_head_idx = self.mha_router._select_heads(router_inputs)
            hidden_states = self.mixer(
                hidden_states,
                batch_head_idx=batch_head_idx,
                **mixer_kwargs
            )
            
            main_stream.record_event(self.mha_event)

        # Now we unify after both are done, then do the next steps
        with torch.cuda.stream(main_stream):
            # Wait on router & MHA
            curr_stream.wait_stream(main_stream)
            main_stream.wait_event(self.mha_event)

            # normal residual / layernorm
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]

            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(
                        residual.to(dtype=self.norm2.weight.dtype)
                    )
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                hidden_states.shape[:-1],
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                            )
                        )
                    if hidden_states.shape != residual.shape:
                        hidden_states = hidden_states.view(residual.shape)
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=residual,
                        eps=self.norm2.eps,
                        dropout_p=self.dropout2.p if self.training else 0.0,
                        rowscale=rowscale2,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        is_rms_norm=isinstance(self.norm2, RMSNorm),
                    )
                
                # hidden_states = self.mlp(hidden_states, index_vec=test_index_vec, index_size=test_index_size)
                if self.skip_mlp_router:
                    hidden_states = self.mlp(hidden_states, index_vec=None)
                else:
                    curr_stream.wait_stream(self.sparse_stream)
                    main_stream.wait_event(self.mlp_event)
                    hidden_states = self.mlp(hidden_states, index_vec=index_vec)
                curr_stream.wait_stream(main_stream)
                curr_stream.wait_stream(self.sparse_stream)

        return hidden_states, residual
    
    def tp_decode_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_subset=None, mixer_kwargs=None):
        """
        Tensor Parallel Decode Forward
    
        """
        
        curr_stream = torch.cuda.current_stream()
        self.sparse_stream.wait_stream(curr_stream)
        # self.main_stream.wait_stream(curr_stream)
        
        router_inputs = hidden_states.squeeze(1)  # shape (batch_size, dim) 
        
        if self.mlp_topk > 0.8 * self.total_neurons:
            self.skip_mlp_router = True
        else:
            self.skip_mlp_router = False
        
        # attention router is synchronous
        batch_head_idx = self.mha_router._select_heads(router_inputs)
        
        # mlp router is asynchronous
        if not self.skip_mlp_router:
            with torch.cuda.stream(self.sparse_stream):
                index_vec = self.mlp_router._select_neurons_topk(router_inputs, topk = self.mlp_topk)
                self.sparse_stream.record_event(self.mlp_event)      
        
        hidden_states = self.mixer(hidden_states, **mixer_kwargs, batch_head_idx=batch_head_idx)
            
        if mixer_subset is not None:
            residual = residual[:, mixer_subset]

        if not isinstance(self.mlp, nn.Identity):
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path2(self.dropout2(hidden_states))
                if dropped.shape != residual.shape:
                    dropped = dropped.view(residual.shape)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                if hidden_states.shape != residual.shape:
                    hidden_states = hidden_states.view(residual.shape)
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    eps=self.norm2.eps,
                    dropout_p=self.dropout2.p if self.training else 0.0,
                    rowscale=rowscale2,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            
            # curr_stream.wait_stream(self.sparse_stream)
            if self.skip_mlp_router:
                hidden_states = self.mlp(hidden_states, index_vec=None)
            else:
                curr_stream.wait_event(self.mlp_event)
                hidden_states = self.mlp(hidden_states, index_vec=index_vec)
        
        return hidden_states, residual

    def attn_sparse_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_subset=None, mixer_kwargs=None):
        """
        Decode Forward with Sparse Attention Router
        """
        
        # We want to run MHA & mlp_router in parallel on different streams
        router_inputs = hidden_states.squeeze(1)  # shape (batch_size, dim)
        
        batch_head_idx = self.mha_router._select_heads(router_inputs)
        
        # print(f"hidden_states shape: {hidden_states.shape}")
        # print(f"hidden states: {hidden_states}")
        hidden_states = self.mixer(hidden_states, batch_head_idx=batch_head_idx, **mixer_kwargs)

        # normal residual / layernorm
        if mixer_subset is not None:
            residual = residual[:, mixer_subset]

        if not isinstance(self.mlp, nn.Identity):
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(
                    residual.to(dtype=self.norm2.weight.dtype)
                )
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(hidden_states.shape[:-1], device=hidden_states.device, dtype=hidden_states.dtype,)
                    )
                if hidden_states.shape != residual.shape:
                    hidden_states = hidden_states.view(residual.shape)
                hidden_states, residual = layer_norm_fn(hidden_states, self.norm2.weight, self.norm2.bias, residual=residual, 
                                                        eps=self.norm2.eps, dropout_p=self.dropout2.p if self.training else 0.0, 
                                                        rowscale=rowscale2, prenorm=True, residual_in_fp32=self.residual_in_fp32, 
                                                        is_rms_norm=isinstance(self.norm2, RMSNorm),)
            
            # hidden_states = self.mlp(hidden_states, index_vec=test_index_vec, index_size=test_index_size)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
    
    def mlp_sparse_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_subset=None, mixer_kwargs=None):
        """ Single GPU Decode Forward

        Args:
            hidden_states (Tensor): _description_
            residual (Optional[Tensor], optional): _description_. Defaults to None.
            mixer_subset (_type_, optional): _description_. Defaults to None.
        """
        curr_stream = torch.cuda.current_stream()
        
        # We want to run MHA & mlp_router in parallel on different streams
        router_inputs = hidden_states.squeeze(1)  # shape (batch_size, dim)
        self.main_stream.wait_stream(curr_stream)
        self.sparse_stream.wait_stream(curr_stream)
        main_stream = self.main_stream
        
        # if mlp_topk > th * total_neurons, skip mlp router
        
        if self.mlp_topk > 0.8 * self.total_neurons:
            self.skip_mlp_router = True
        else:
            self.skip_mlp_router = False

        # [Sparse stream]  mlp_router
        if not self.skip_mlp_router:
            with torch.cuda.stream(self.sparse_stream):  
                index_vec = self.mlp_router._select_neurons_topk(router_inputs, topk = self.mlp_topk)
                self.sparse_stream.record_event(self.mlp_event)

        # [Main stream]  MHA
        with torch.cuda.stream(main_stream):
            # batch_head_idx = self.mha_router._select_heads(router_inputs)
            hidden_states = self.mixer(
                hidden_states,
                batch_head_idx=None,
                **mixer_kwargs
            )
            
            main_stream.record_event(self.mha_event)

        # Now we unify after both are done, then do the next steps
        with torch.cuda.stream(main_stream):
            # Wait on router & MHA
            curr_stream.wait_stream(main_stream)
            main_stream.wait_event(self.mha_event)

            # normal residual / layernorm
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]

            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(
                        residual.to(dtype=self.norm2.weight.dtype)
                    )
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                hidden_states.shape[:-1],
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                            )
                        )
                    if hidden_states.shape != residual.shape:
                        hidden_states = hidden_states.view(residual.shape)
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=residual,
                        eps=self.norm2.eps,
                        dropout_p=self.dropout2.p if self.training else 0.0,
                        rowscale=rowscale2,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        is_rms_norm=isinstance(self.norm2, RMSNorm),
                    )
                
                # hidden_states = self.mlp(hidden_states, index_vec=test_index_vec, index_size=test_index_size)
                if self.skip_mlp_router:
                    hidden_states = self.mlp(hidden_states, index_vec=None)
                else:
                    curr_stream.wait_stream(self.sparse_stream)
                    main_stream.wait_event(self.mlp_event)
                    hidden_states = self.mlp(hidden_states, index_vec=index_vec)
                curr_stream.wait_stream(main_stream)
                curr_stream.wait_stream(self.sparse_stream)

        return hidden_states, residual

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        mixer_subset=None,
        mixer_kwargs=None,
        mlp_topk=None,
        attn_topk=None,
    ):
        """
        This forward pass includes concurrency logic in the decode branch. 
        If you're capturing with a CUDA graph, the concurrency (two-stream usage) must be 
        inside the captured region so that the replay reproduces the parallel streams.
        """
        
        # simulation values
        if mlp_topk is not None:
            self.mlp_topk = mlp_topk
                
        if attn_topk is not None:
            self.mha_router.topk = attn_topk

        if mixer_kwargs is None:
            mixer_kwargs = {"inference_params": None}
        else:
            # Ensure 'inference_params' key exists
            if "inference_params" not in mixer_kwargs:
                mixer_kwargs["inference_params"] = None

        if self.prenorm:
            # --- 1) Prenorm’s dropout/add/layernorm
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                # fused dropout + add + layernorm
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                if residual is not None and hidden_states.shape != residual.shape:
                    hidden_states = hidden_states.view(residual.shape)
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm1.weight,
                    self.norm1.bias,
                    residual=residual,
                    eps=self.norm1.eps,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    rowscale=rowscale1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm1, RMSNorm),
                )

            if mixer_subset is not None:
                mixer_kwargs["mixer_subset"] = mixer_subset

            # Check if we are in the prefill or decode stage
            prefill_stage = (
                mixer_kwargs["inference_params"] is None
                or mixer_kwargs["inference_params"].seqlen_offset == 0
            )

            if prefill_stage:
                # --- 2) Prefill stage (no concurrency): just do normal forward
                hidden_states, residual = self.prefill_forward(hidden_states, residual, mixer_kwargs, mixer_subset)

            else:
                # --- 3) Decode stage:
                if self.mlp_router is None:
                    # decode stage with only attention router, works with both single gpu and tensor parallel
                    hidden_states, residual = self.attn_sparse_forward(hidden_states, residual, mixer_subset, mixer_kwargs)
                else:
                    if not self.use_tensor_parallel:
                        if self.mha_router is None:
                            # decode stage with mlp routers (opt models and single gpu)
                            hidden_states, residual = self.mlp_sparse_forward(hidden_states, residual, mixer_subset, mixer_kwargs)
                        else:
                            # decode stage with mlp and attention routers (opt models and single gpu)
                            hidden_states, residual = self.decode_forward(hidden_states, residual, mixer_subset, mixer_kwargs)
                    else:
                        # uses both mlp and attention routers in tensor parallel
                        hidden_states, residual = self.tp_decode_forward(hidden_states, residual, mixer_subset, mixer_kwargs)
                    
            return hidden_states, residual

        else:
            # post-norm architecture not implemented here
            raise NotImplementedError


# class SelectBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         mixer_cls=None,
#         mlp_cls=None,
#         mlp_router=None,
#         mha_router=None,
#         norm_cls=nn.LayerNorm,
#         dropout_cls=nn.Dropout,
#         prenorm=True,
#         resid_dropout1=0.0,
#         resid_dropout2=0.0,
#         drop_path1=0.0,
#         drop_path2=0.0,
#         fused_dropout_add_ln=False,
#         return_residual=False,
#         residual_in_fp32=False,
#         sequence_parallel=False,
#         mark_shared_params=False,
#     ):
#         """
#         For prenorm=True, this Block has a slightly different structure compared to a regular
#         prenorm Transformer block.

#         The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
#         Here we do: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, etc.

#         If you want to do concurrency with CUDA graphs, your shapes must remain fixed 
#         (batch_size, seq_len, etc.) across captures and replays. Also avoid any operations 
#         that cause dynamic shape changes or memory allocations.
#         """
#         super().__init__()
#         self.prenorm = prenorm
#         self.fused_dropout_add_ln = fused_dropout_add_ln
#         self.return_residual = return_residual
#         self.residual_in_fp32 = residual_in_fp32
#         if self.residual_in_fp32:
#             assert self.prenorm, "residual_in_fp32 is only compatible with prenorm=True"
        
#         assert mixer_cls is not None and mlp_cls is not None, (
#             "mixer_cls and mlp_cls cannot be None in SelectBlock"
#         )
            
#         # MHA & MLP submodules
#         self.mixer = mixer_cls(dim)
#         self.dropout1 = dropout_cls(resid_dropout1)
#         self.drop_path1 = StochasticDepth(drop_path1, mode="row")
#         self.norm1 = norm_cls(dim)
#         self.mlp = mlp_cls(dim)
        
#         # Routers
#         self.mlp_router = mlp_router(dim)
#         self.mha_router = mha_router(dim)

#         if not isinstance(self.mlp, nn.Identity):
#             self.dropout2 = dropout_cls(resid_dropout2)
#             self.drop_path2 = StochasticDepth(drop_path2, mode="row")
#             self.norm2 = norm_cls(dim)

#         if self.fused_dropout_add_ln:
#             assert layer_norm_fn is not None, "Triton layer_norm_fn not installed"
#             assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(self.dropout1, nn.Dropout)

#         # Mark the norm parameters for sequence parallel / shared params if needed
#         if sequence_parallel:
#             for p in self.norm1.parameters():
#                 p._sequence_parallel = True
#             if hasattr(self, "norm2"):
#                 for p in self.norm2.parameters():
#                     p._sequence_parallel = True
#         if mark_shared_params:
#             for p in self.norm1.parameters():
#                 p._shared_params = True
#             if hasattr(self, "norm2"):
#                 for p in self.norm2.parameters():
#                     p._shared_params = True

#         self.mlp_topk = None
#         self.skip_mlp_router = False
#         self.skip_attn_router = False
        
#         # We'll use an extra stream for concurrency
#         self.sparse_stream = torch.cuda.Stream(device="cuda", priority=0)
#         self.main_stream = torch.cuda.Stream(device="cuda", priority=-5)
#         # We'll record events to coordinate concurrency
#         self.mha_event = torch.cuda.Event(enable_timing=False, blocking=False)
#         self.mlp_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
#         self.use_tensor_parallel = mark_shared_params
        
#         if self.use_tensor_parallel:
#             # TODO: save the routers in the mixer and mlp classes
#             # save the stream and events in the mixer and mlp classes
#             pass

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

#     def prefill_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_kwargs=None, mixer_subset=None):
#         hidden_states = self.mixer(hidden_states, **mixer_kwargs)
        
#         if mixer_subset is not None:
#             residual = residual[:, mixer_subset]

#         if not isinstance(self.mlp, nn.Identity):
#             if not self.fused_dropout_add_ln:
#                 dropped = self.drop_path2(self.dropout2(hidden_states))
#                 if dropped.shape != residual.shape:
#                     dropped = dropped.view(residual.shape)
#                 residual = (dropped + residual) if residual is not None else dropped
#                 hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
#                 if self.residual_in_fp32:
#                     residual = residual.to(torch.float32)
#             else:
#                 if self.drop_path2.p == 0 or not self.training:
#                     rowscale2 = None
#                 else:
#                     rowscale2 = self.drop_path2(
#                         torch.ones(
#                             hidden_states.shape[:-1],
#                             device=hidden_states.device,
#                             dtype=hidden_states.dtype,
#                         )
#                     )
#                 if hidden_states.shape != residual.shape:
#                     hidden_states = hidden_states.view(residual.shape)
#                 hidden_states, residual = layer_norm_fn(
#                     hidden_states,
#                     self.norm2.weight,
#                     self.norm2.bias,
#                     residual=residual,
#                     eps=self.norm2.eps,
#                     dropout_p=self.dropout2.p if self.training else 0.0,
#                     rowscale=rowscale2,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     is_rms_norm=isinstance(self.norm2, RMSNorm),
#                 )
#             hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual
    
#     def decode_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_subset=None, mixer_kwargs=None):
#         """ Single GPU Decode Forward

#         Args:
#             hidden_states (Tensor): _description_
#             residual (Optional[Tensor], optional): _description_. Defaults to None.
#             mixer_subset (_type_, optional): _description_. Defaults to None.
#         """
#         curr_stream = torch.cuda.current_stream()
        
#         # We want to run MHA & mlp_router in parallel on different streams
#         router_inputs = hidden_states.squeeze(1)  # shape (batch_size, dim)
#         self.main_stream.wait_stream(curr_stream)
#         self.sparse_stream.wait_stream(curr_stream)
        
#         # We'll do MHA on the "main_stream" and the router on "sparse_stream"
#         main_stream = self.main_stream
#         # In a captured region, each 'with torch.cuda.stream(...)' block
#         # is replayed in concurrency. The shape must remain consistent.

#         # [Sparse stream]  mlp_router
#         if not self.skip_mlp_router:
#             with torch.cuda.stream(self.sparse_stream):  # <-- CHANGED
#                 # index_size, index_vec  = self.mlp_router._select_neurons_cuda_safe(router_inputs)    # need to fix this; make CUDA Graph safe
#                 # vec = self.mlp_router(router_inputs)
#                 index_vec = self.mlp_router._select_neurons_topk(router_inputs, topk = self.mlp_topk)
#                 self.sparse_stream.record_event(self.mlp_event)

#         # [Main stream]  MHA
#         with torch.cuda.stream(main_stream):  # <-- CHANGED
#             batch_head_idx = self.mha_router._select_heads(router_inputs)
#             hidden_states = self.mixer(
#                 hidden_states,
#                 batch_head_idx=batch_head_idx,
#                 # batch_head_idx=None,
#                 **mixer_kwargs
#             )
#             main_stream.record_event(self.mha_event)

#         # Now we unify after both are done, then do the next steps
#         with torch.cuda.stream(main_stream):  # <-- CHANGED
#             # Wait on router & MHA
#             curr_stream.wait_stream(main_stream)
#             main_stream.wait_event(self.mha_event)

#             # normal residual / layernorm
#             if mixer_subset is not None:
#                 residual = residual[:, mixer_subset]

#             if not isinstance(self.mlp, nn.Identity):
#                 if not self.fused_dropout_add_ln:
#                     dropped = self.drop_path2(self.dropout2(hidden_states))
#                     residual = (dropped + residual) if residual is not None else dropped
#                     hidden_states = self.norm2(
#                         residual.to(dtype=self.norm2.weight.dtype)
#                     )
#                     if self.residual_in_fp32:
#                         residual = residual.to(torch.float32)
#                 else:
#                     if self.drop_path2.p == 0 or not self.training:
#                         rowscale2 = None
#                     else:
#                         rowscale2 = self.drop_path2(
#                             torch.ones(
#                                 hidden_states.shape[:-1],
#                                 device=hidden_states.device,
#                                 dtype=hidden_states.dtype,
#                             )
#                         )
#                     if hidden_states.shape != residual.shape:
#                         hidden_states = hidden_states.view(residual.shape)
#                     hidden_states, residual = layer_norm_fn(
#                         hidden_states,
#                         self.norm2.weight,
#                         self.norm2.bias,
#                         residual=residual,
#                         eps=self.norm2.eps,
#                         dropout_p=self.dropout2.p if self.training else 0.0,
#                         rowscale=rowscale2,
#                         prenorm=True,
#                         residual_in_fp32=self.residual_in_fp32,
#                         is_rms_norm=isinstance(self.norm2, RMSNorm),
#                     )

#                 # Finally do MLP with the router's index vector
#                 curr_stream.wait_stream(self.sparse_stream)
#                 main_stream.wait_event(self.mlp_event)
                
#                 # hidden_states = self.mlp(hidden_states, index_vec=test_index_vec, index_size=test_index_size)
#                 if self.skip_mlp_router:
#                     hidden_states = self.mlp(hidden_states, index_vec=None)
#                 else:
#                     hidden_states = self.mlp(hidden_states, index_vec=index_vec)
#                 curr_stream.wait_stream(main_stream)
#                 curr_stream.wait_stream(self.sparse_stream)

#         return hidden_states, residual
    
#     def tp_decode_forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, mixer_subset=None, mixer_kwargs=None):
#         """
#         Tensor Parallel Decode Forward
        
#         Args:
#             hidden_states (Tensor): _description_
#             residual (Optional[Tensor], optional): _description_. Defaults to None.
#             mixer_subset (_type_, optional): _description_. Defaults to None.
#         """
#         # TODO: need to add routing 
        
#         hidden_states = self.mixer(hidden_states, **mixer_kwargs)
        
#         if mixer_subset is not None:
#             residual = residual[:, mixer_subset]

#         if not isinstance(self.mlp, nn.Identity):
#             if not self.fused_dropout_add_ln:
#                 dropped = self.drop_path2(self.dropout2(hidden_states))
#                 if dropped.shape != residual.shape:
#                     dropped = dropped.view(residual.shape)
#                 residual = (dropped + residual) if residual is not None else dropped
#                 hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
#                 if self.residual_in_fp32:
#                     residual = residual.to(torch.float32)
#             else:
#                 if self.drop_path2.p == 0 or not self.training:
#                     rowscale2 = None
#                 else:
#                     rowscale2 = self.drop_path2(
#                         torch.ones(
#                             hidden_states.shape[:-1],
#                             device=hidden_states.device,
#                             dtype=hidden_states.dtype,
#                         )
#                     )
#                 if hidden_states.shape != residual.shape:
#                     hidden_states = hidden_states.view(residual.shape)
#                 hidden_states, residual = layer_norm_fn(
#                     hidden_states,
#                     self.norm2.weight,
#                     self.norm2.bias,
#                     residual=residual,
#                     eps=self.norm2.eps,
#                     dropout_p=self.dropout2.p if self.training else 0.0,
#                     rowscale=rowscale2,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     is_rms_norm=isinstance(self.norm2, RMSNorm),
#                 )
#             hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual

#     def forward(
#         self,
#         hidden_states: Tensor,
#         residual: Optional[Tensor] = None,
#         mixer_subset=None,
#         mixer_kwargs=None,
#     ):
#         """
#         This forward pass includes concurrency logic in the decode branch. 
#         If you're capturing with a CUDA graph, the concurrency (two-stream usage) must be 
#         inside the captured region so that the replay reproduces the parallel streams.
#         """
        

#         if mixer_kwargs is None:
#             mixer_kwargs = {"inference_params": None}
#         else:
#             # Ensure 'inference_params' key exists
#             if "inference_params" not in mixer_kwargs:
#                 mixer_kwargs["inference_params"] = None

#         if self.prenorm:
#             # --- 1) Prenorm’s dropout/add/layernorm
#             if not self.fused_dropout_add_ln:
#                 dropped = self.drop_path1(self.dropout1(hidden_states))
#                 residual = (dropped + residual) if residual is not None else dropped
#                 hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
#                 if self.residual_in_fp32:
#                     residual = residual.to(torch.float32)
#             else:
#                 # fused dropout + add + layernorm
#                 if self.drop_path1.p == 0 or not self.training:
#                     rowscale1 = None
#                 else:
#                     rowscale1 = self.drop_path1(
#                         torch.ones(
#                             hidden_states.shape[:-1],
#                             device=hidden_states.device,
#                             dtype=hidden_states.dtype,
#                         )
#                     )
#                 if residual is not None and hidden_states.shape != residual.shape:
#                     hidden_states = hidden_states.view(residual.shape)
#                 hidden_states, residual = layer_norm_fn(
#                     hidden_states,
#                     self.norm1.weight,
#                     self.norm1.bias,
#                     residual=residual,
#                     eps=self.norm1.eps,
#                     dropout_p=self.dropout1.p if self.training else 0.0,
#                     rowscale=rowscale1,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     is_rms_norm=isinstance(self.norm1, RMSNorm),
#                 )

#             if mixer_subset is not None:
#                 mixer_kwargs["mixer_subset"] = mixer_subset

#             # Check if we are in the prefill or decode stage
#             prefill_stage = (
#                 mixer_kwargs["inference_params"] is None
#                 or mixer_kwargs["inference_params"].seqlen_offset == 0
#             )

#             if prefill_stage:
#                 # --- 2) Prefill stage (no concurrency): just do normal forward
#                 hidden_states, residual = self.prefill_forward(hidden_states, residual, mixer_kwargs, mixer_subset)

#             else:
#                 # # --- 3) Decode stage:
#                 if not self.use_tensor_parallel:
#                     hidden_states, residual = self.decode_forward(hidden_states, residual, mixer_subset, mixer_kwargs)
#                 else:
#                     # routing is slightly different in tensor parallel; we overlap the router with allreduce
#                     hidden_states, residual = self.tp_decode_forward(hidden_states, residual, mixer_subset)
                    
#             return hidden_states, residual

#         else:
#             # post-norm architecture not implemented here
#             raise NotImplementedError
