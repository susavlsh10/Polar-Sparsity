import torch
import torch.nn as nn
try:
    import AmpSelGemm
except ImportError:
    AmpSelGemm = None
    
from HybridTensor.triton.gather_gemm_col import gather_matmul_col
from HybridTensor.triton.gather_gemm_row import gather_matmul_row
from HybridTensor.utils.utils import arg_parser, sparse_index, create_results_directory 
from HybridTensor.utils.profiling import benchmark_mlp_fwd, generate_index_sizes, save_results_to_csv, plot_results

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm  # For progress bars

# implement standard MLP block with ReLU activation
class StandardMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False, device='cuda'):
        super(StandardMLPBlock, self).__init__()
        hidden_features = hidden_features or in_features*4
        
        # this is stored in correct order; don't need to transpose for the CUTLASS kernel
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device, dtype=torch.float16)
        
        # this is stored in row major; need to transpose for the CUTLASS kernel
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias, device=device, dtype=torch.float16)
        self.relu = nn.ReLU()

    def forward(self, x):
        # fc1 : d x (4d)
        out = self.fc1(x)
        # B x (4d)
        out = self.relu(out)
        
        #fc2: (4d) x d
        out = self.fc2(out)
        return out
    
    def sparsify(self, zero_index):
        self.fc1.weight.data[zero_index, :] = 0.0
        self.fc2.weight.data[:, zero_index] = 0.0
        
class SelectiveMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False, requires_grad = False, device='cuda', dtype=torch.float16, activation='relu'):
        super(SelectiveMLP, self).__init__()
        if hidden_features is None:
            hidden_features = in_features*4
            
        factory_kwargs = {'device': torch.device(device), 'dtype': dtype}
        self.fc1_w = torch.empty((hidden_features, in_features), requires_grad=requires_grad, **factory_kwargs)
        self.fc2_w = torch.empty((hidden_features, in_features), requires_grad=requires_grad, **factory_kwargs)
        self.act = nn.ReLU()
        
    def forward(self, x, index_vec):
        index_size = index_vec.size(0)
        #AmpSelGemm.run(A=A, B=B_col_major, index_vec= index_vec, M= M, N=index_size, K=K, index_size=index_size)
        out = AmpSelGemm.run_col(A=x, B=self.fc1_w, index_vec= index_vec, M = x.size(0), N = index_size, K = self.fc1_w.size(1), index_size=index_size)
        out = self.act(out)   # need to fuse this with fc1 in the next iteration
        out = AmpSelGemm.run_row1(A=out, B=self.fc2_w, index_vec= index_vec, M = x.size(0), N = self.fc2_w.size(1), K = index_size, index_size=index_size)
        return out
    
    def load_from_MLP(self, mlp):
        self.fc1_w = mlp.fc1.weight  
        self.fc2_w = mlp.fc2.weight.t().contiguous()  
        return self
    
class SelectiveMLPTriton(SelectiveMLP):
    def __init__(self, in_features, hidden_features=None, bias=False, requires_grad = False, device='cuda', dtype=torch.float16, activation='relu'):
        super(SelectiveMLPTriton, self).__init__(in_features, hidden_features, bias, requires_grad, device, dtype, activation)
        self.activation = activation

    def forward(self, x, index_vec):
        out = gather_matmul_col(x, self.fc1_w, index_vec, activations=self.activation)
        out = gather_matmul_row(out, self.fc2_w, index_vec)
        return out

def SelectiveMLPFunc(x, fc1_w, fc2_w, index_vec, bias1 = None, bias2 = None, activation='relu'):
    out = gather_matmul_col(x, fc1_w, index_vec, bias = bias1, activations=activation)
    out = gather_matmul_row(out, fc2_w, index_vec, bias = bias2)
    return out

def profile_mlps(args, index_size):
    # Create standard MLP block
    standardmlp = StandardMLPBlock(args.in_features, args.hidden_features)
    if AmpSelGemm is not None:
        selectiveMLP = SelectiveMLP(args.in_features, args.hidden_features).load_from_MLP(standardmlp)
    selectiveMLPTriton = SelectiveMLPTriton(args.in_features, args.hidden_features).load_from_MLP(standardmlp)
    
    # Test input
    x = torch.randn(args.batch_size, args.in_features, dtype=torch.float16, device='cuda')
    index_vec, zero_index = sparse_index(index_size, args.hidden_features or args.in_features*4)
    standardmlp.sparsify(zero_index)
    
    # Measure execution time
    std_out, standardmlp_time = benchmark_mlp_fwd(x, standardmlp, index_vec=None, iterations=args.iterations, print_result=False)
    if AmpSelGemm is not None:
        cutlass_out, selectiveMLP_time = benchmark_mlp_fwd(x, selectiveMLP, index_vec=index_vec, iterations=args.iterations, print_result=False)
    triton_out, selectiveMLPTriton_time = benchmark_mlp_fwd(x, selectiveMLPTriton, index_vec=index_vec, iterations=args.iterations, print_result=False)
    
    # Optionally check results
    if args.check_results:
        print('Standard MLP output:', std_out[0])
        if AmpSelGemm is not None:
            print('Selective MLP Cutlass output:', cutlass_out)
        print('Selective MLP Triton output:', triton_out)
    
    # Calculate speedups
    triton_speedup = standardmlp_time / selectiveMLPTriton_time if selectiveMLPTriton_time > 0 else float('inf')
    if AmpSelGemm is not None:
        cutlass_speedup = standardmlp_time / selectiveMLP_time if selectiveMLP_time > 0 else float('inf')
    
    return {
        'index_size': index_size,
        'standard_time': standardmlp_time,
        'selective_cutlass_time': selectiveMLP_time,
        'selective_triton_time': selectiveMLPTriton_time,
        'cutlass_speedup': cutlass_speedup,
        'triton_speedup': triton_speedup
    }
    
def run_profiling_over_index_sizes(args, index_sizes):
    results = []
    for size in tqdm(index_sizes, desc="Profiling MLPs"):
        result = profile_mlps(args, size)
        results.append(result)
    return pd.DataFrame(results)


'''
if __name__ == '__main__':
    args = arg_parser()

    index_sizes = generate_index_sizes(args.hidden_features)
    
    # create standard MLP block
    standardmlp = StandardMLPBlock(args.in_features, args.hidden_features)
    selectiveMLP = SelectiveMLP(args.in_features, args.hidden_features).load_from_MLP(standardmlp)
    selectiveMLPTriton = SelectiveMLPTriton(args.in_features, args.hidden_features).load_from_MLP(standardmlp)
    
    
    # test input
    x = torch.randn(args.batch_size, args.in_features, dtype=torch.float16, device='cuda')
    index_vec, zero_index = sparse_index(args.index_size, args.hidden_features or args.in_features*4)
    standardmlp.sparsify(zero_index)
    
    # measure execution time
    std_out, standardmlp_time = benchmark_mlp_fwd(x, standardmlp, index_vec= None, iterations=args.iterations, print_result=True)
    cutlass_out, selectiveMLP_time = benchmark_mlp_fwd(x, selectiveMLP, index_vec= index_vec, iterations=args.iterations, print_result=True)
    triton_out, selectiveMLPTriton_time = benchmark_mlp_fwd(x, selectiveMLPTriton, index_vec= index_vec, iterations=args.iterations, print_result=True)
    
    if args.check_results:
        print('Standard MLP output:', std_out[0])
        print('Selective MLP Cutlass output:', cutlass_out)
        print('Selective MLP Triton output:', triton_out)
    
    triton_speedup = standardmlp_time/selectiveMLPTriton_time
    cutlass_speedup = standardmlp_time/selectiveMLP_time
    
    print(f"Speedup of Cutlass implementation over standard MLP: {cutlass_speedup}")
    print(f"Speedup of Triton implementation over standard MLP: {triton_speedup}")

'''
    
if __name__ == '__main__':
    args = arg_parser()
    
    print(f"Profiling MLPs")
    # Results directory
    results_dir = create_results_directory(args.results_dir)
    
    # Define the range of index sizes you want to profile
    index_sizes = generate_index_sizes(args.hidden_features)
    
    # Run profiling over different index sizes
    profiling_results = run_profiling_over_index_sizes(args, index_sizes)
    
    # Save the results to a CSV file
    save_results_to_csv(profiling_results, filename_prefix='mlp_profiling_results', results_dir=results_dir)
    
    # Plot the results
    plot_results(profiling_results, output_prefix='mlp_profiling', results_dir=results_dir)
    
    # Optionally, print the DataFrame
    if args.check_results:
        print(profiling_results)