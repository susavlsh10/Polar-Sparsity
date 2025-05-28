import torch

def select_heads(heads, top_k):
    _, top_indices = torch.topk(heads, top_k, dim=1)
    return top_indices

def select_neurons(neurons, th):
    activated = (neurons > th).sum(dim = 0)
    index_vec = activated.nonzero().flatten()
    return index_vec

class FusedRouter(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dim = 1024, mlp_th = 0.5, attn_top_k = 0.3, device='cuda', dtype=torch.float16):
        super(FusedRouter, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.total_neurons = embed_dim * 4
        
        self.mlp_th = mlp_th
        self.top_k = int(num_heads * attn_top_k)
        
        self.fc1 = torch.nn.Linear(embed_dim, dim, bias=False, device=device, dtype=dtype)
        self.layer_norm1 = torch.nn.LayerNorm(dim, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(dim, self.total_neurons + num_heads, bias=False, device=device, dtype=dtype)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.layer_norm1(out)
        out = self.fc2(out)
        neurons = out[:, :self.total_neurons]
        heads = out[:, self.total_neurons:]
        
        return neurons, heads
    
    def select_neurons_heads(self, x):
        neurons, heads = self(x)
        selected_neurons = select_neurons(neurons, self.mlp_th)
        selected_heads = select_heads(heads, self.top_k)
        return selected_neurons, selected_heads
    
class HybridFusedRouter(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim = 1024, mha_dim = 128, mlp_th = 0.5, attn_top_k = 0.3, device='cuda', dtype=torch.float16):
        super(HybridFusedRouter, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.total_neurons = embed_dim * 4
        self.mlp_dim = mlp_dim
        self.mha_dim = mha_dim
        
        self.mlp_th = mlp_th
        self.top_k = int(num_heads * attn_top_k)
        
        self.fc1 = torch.nn.Linear(embed_dim, mlp_dim + mha_dim, bias=False, device=device, dtype=dtype)
        self.layer_norm1 = torch.nn.LayerNorm(mlp_dim, device=device, dtype=dtype)
        self.activation = torch.nn.ReLU()
        
        self.fc2_mlp = torch.nn.Linear(mlp_dim, self.total_neurons, bias=False, device=device, dtype=dtype)
        self.fc2_mha = torch.nn.Linear(mha_dim, num_heads, bias=False, device=device, dtype=dtype)
        
        # cache for neurons
        self.mlp_neurons = None
        
    def forward(self, x):
        out = self.fc1(x)
        mlp_out, mha_out = out[:, :self.mlp_dim], out[:, self.mlp_dim:]
        
        # mlp router
        neurons = self.layer_norm1(mlp_out)
        neurons = self.fc2_mlp(mlp_out)
        
        # mha router
        heads = self.activation(mha_out)
        heads = self.fc2_mha(mha_out)
        
        return neurons, heads
    
    def select_heads_(self, x):
        out = self.fc1(x)
        mlp_out, mha_out = out[:, :self.mlp_dim], out[:, self.mlp_dim:]
        self.mlp_neurons = mlp_out
        
        # mha router
        heads = self.activation(mha_out)
        heads = self.fc2_mha(mha_out)
        
        selected_heads = select_heads(heads, self.top_k)
        return selected_heads
        
    def select_neurons_(self, x):
        neurons = self.layer_norm1(self.mlp_neurons)
        neurons = self.fc2_mlp(neurons)
        selected_neurons = select_neurons(neurons, self.mlp_th)
        return selected_neurons
        
    
    def select_neurons_heads(self, x):
        neurons, heads = self(x)
        selected_neurons = select_neurons(neurons, self.mlp_th)
        selected_heads = select_heads(heads, self.top_k)
        return selected_neurons, selected_heads

