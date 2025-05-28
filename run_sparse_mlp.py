# python run_sparse_mlp.py --in_features 8192 --batch_size 16 --index_size 8192


import torch
from HybridTensor.modules.SelectiveMLP import SelectiveMLP
from HybridTensor.utils.profiling import cuda_profiler
from HybridTensor.utils.utils import arg_parser, sparse_index
import torch.nn as nn
import torch.nn.functional as F

# standard MLP implementation
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

if __name__ == "__main__":
    args = arg_parser()
    
    bias = True if args.bias > 0 else False
    
    in_features = args.in_features
    hidden_features = in_features * 4
    out_features = in_features
    activation="relu"
    device = torch.device("cuda")
    
    sparse_mlp = SelectiveMLP(
        in_features, hidden_features, out_features, activation="relu", use_heuristic=False, bias1=bias, bias2=bias, device=device, dtype=torch.float16
    )
    
    activation_fn = F.relu if activation == "relu" else F.gelu
    dense_mlp = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features, bias1=bias, bias2=bias, activation=activation_fn, device=device, dtype=torch.float16)


    # Create random input tensor
    x = torch.randn(args.batch_size, args.in_features, device="cuda", dtype=torch.float16)
    index_vec, _ = sparse_index(args.index_size, args.in_features*4)

    # dense mlp time 
    dense_mlp_out, dense_mlp_time = cuda_profiler(dense_mlp, x)
    
    # sparse mlp time
    sparse_mlp_out, sparse_mlp_time = cuda_profiler(sparse_mlp, x, index_vec)
    print(f"Dense MLP time: {dense_mlp_time:.4f} ms")
    print(f"Sparse MLP time: {sparse_mlp_time:.4f} ms")
    print(f"Speedup: {dense_mlp_time/sparse_mlp_time:.2f}x")
    
    