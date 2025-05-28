# python -m HybridTensor.triton.heuristics.autotune_configs --in_features 4096 --mode auto

import os
import re
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
from HybridTensor.utils.utils import arg_parser


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# Updated regex to capture all parameters: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
# num_warps, num_ctas, num_stages, maxnreg
config_line_pattern = re.compile(
    r"best config selected:\s*BLOCK_SIZE_M:\s*(\d+),\s*BLOCK_SIZE_N:\s*(\d+),\s*BLOCK_SIZE_K:\s*(\d+),\s*GROUP_SIZE_M:\s*(\d+),\s*num_warps:\s*(\d+),\s*num_ctas:\s*(\d+),\s*num_stages:\s*(\d+),\s*maxnreg:\s*(\w+);"
)

args_line_pattern = re.compile(
    r"args: Namespace\(batch_size=(\d+), hidden_features=(\d+), in_features=(\d+), seq_len=(\d+), index_size=(\d+),"
)

kernel_name_pattern = re.compile(
    r"Triton autotuning for function (\w+) finished after"
)

def run_offline_autotune_and_record_1(in_features, mode = "row"):
    '''
    Records the best configurations for the gather_gemm_{row/col} kernel for different batch sizes and index sizes.
    Old function, does not append to existing configs.
    '''
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64]
    # batch_sizes = [8]
    
    
    # Index sizes from 10% to 100% of hidden_features in 10% increments
    num_neurons = in_features * 4   # hidden_features
    index_sizes = [int(num_neurons * pct / 100) for pct in range(5, 101, 5)]
    # index_sizes = [1024, 4096, 8192]

    test_cases = []
    for bs in batch_sizes:
        for idx_size in index_sizes:
            test_cases.append((bs, in_features, idx_size))

    best_configs = {}
    kernel_name = None

    for (batch_size, in_features, index_size) in tqdm(test_cases, desc="Autotuning cases"):
        if mode == "row":
            cmd = [
                "python", "-m", "HybridTensor.triton.gather_gemm_row",
                "--batch_size", str(batch_size),
                "--in_features", str(in_features),
                "--index_size", str(index_size)
            ]
        elif mode == "col":
            cmd = [
                "python", "-m", "HybridTensor.triton.gather_gemm_col",
                "--batch_size", str(batch_size),
                "--in_features", str(in_features),
                "--index_size", str(index_size)
            ]
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # M = K = N = idx_size = None
        M = batch_size
        K = in_features
        N = in_features * 4
        idx_size = index_size
        chosen_config = None
        lines = result.stdout.split("\n")

        # Extract M, K, N, idx_size from arguments line
        # for line in lines:
        #     args_match = args_line_pattern.search(line)
        #     if args_match:
        #         parsed_batch_size = int(args_match.group(1))
        #         parsed_hidden_features = int(args_match.group(2))
        #         parsed_in_features = int(args_match.group(3))
        #         parsed_seq_len = int(args_match.group(4))
        #         parsed_index_size = int(args_match.group(5))

        #         # M = parsed_batch_size
        #         # K = parsed_in_features
        #         # N = in_features * 4
        #         M = batch_size
        #         K = in_features
        #         N = in_features * 4
        #         idx_size = index_size
        #         # N = parsed_hidden_features
        #         # idx_size = parsed_index_size
        #         break

        # Extract kernel name and config details
        for line in lines:
            if kernel_name is None:
                kn_match = kernel_name_pattern.search(line)
                if kn_match:
                    kernel_name = kn_match.group(1)

            match = config_line_pattern.search(line)
            if match:
                BLOCK_SIZE_M = int(match.group(1))
                BLOCK_SIZE_N = int(match.group(2))
                BLOCK_SIZE_K = int(match.group(3))
                GROUP_SIZE_M = int(match.group(4))
                num_warps = int(match.group(5))
                num_ctas = int(match.group(6))
                num_stages = int(match.group(7))
                maxnreg_str = match.group(8)
                # Convert maxnreg to None if it's the word 'None'
                maxnreg = None if maxnreg_str == "None" else maxnreg_str

                chosen_config = {
                    "BLOCK_SIZE_M": BLOCK_SIZE_M,
                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                    "BLOCK_SIZE_K": BLOCK_SIZE_K,
                    "GROUP_SIZE_M": GROUP_SIZE_M,
                    "num_warps": num_warps,
                    "num_ctas": num_ctas,
                    "num_stages": num_stages,
                    "maxnreg": maxnreg
                }

        if chosen_config is None:
            print(f"Warning: Could not find chosen config for batch_size={batch_size}, in_features={in_features}, index_size={index_size}")
        else:
            best_configs[(M, K, N, idx_size)] = chosen_config
            print(f"Best config for batch_size={batch_size}, in_features={in_features}, index_size={index_size}: {chosen_config}")

    if kernel_name is None:
        kernel_name = "unknown_kernel"

    # Convert tuple keys to string
    saveable_configs = {str(k): v for k, v in best_configs.items()}
    output_path = Path(f"configs/gemm/best_configs_{kernel_name}_{in_features}.json")
    with open(output_path, "w") as f:
        json.dump(saveable_configs, f, indent=4)

    print(f"Best configs saved to {output_path}")

def run_offline_autotune_and_record(in_features, mode="row"):
    """
    Records the best configurations for the gather_gemm_{row/col} kernel
    for different batch sizes and index sizes.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]
    # batch_sizes = [1]
    
    
    # batch_sizes = [2]
    num_neurons = in_features * 4  # hidden_features
    index_sizes = [int(num_neurons * pct / 100) for pct in range(5, 101, 5)]

    test_cases = [(bs, in_features, idx_size) for bs in batch_sizes for idx_size in index_sizes]

    # Determine output file path. If file exists, load previous configs.
    # kernel_name might be unknown initially; use a temporary kernel tag for filename.
    temp_kernel_tag = "matmul_gather_kernel_" + mode
    output_dir = Path("configs/gemm")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"best_configs_{temp_kernel_tag}_{in_features}.json"
    
    best_configs = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            best_configs = json.load(f)
        print(f"Loaded existing configs from {output_path}")

    kernel_name = None

    for (batch_size, in_features, index_size) in tqdm(test_cases, desc="Autotuning cases"):
        # Define test key and check if exists
        M = batch_size
        K = in_features
        N = in_features * 4
        key_str = str((M, K, N, index_size))
        if key_str in best_configs:
            print(f"Skipping batch_size={batch_size}, in_features={in_features}, index_size={index_size} (config exists)")
            continue

        if mode == "row":
            cmd = [
                "python", "-m", "HybridTensor.triton.gather_gemm_row",
                "--batch_size", str(batch_size),
                "--in_features", str(in_features),
                "--index_size", str(index_size)
            ]
        elif mode == "col":
            cmd = [
                "python", "-m", "HybridTensor.triton.gather_gemm_col",
                "--batch_size", str(batch_size),
                "--in_features", str(in_features),
                "--index_size", str(index_size)
            ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        chosen_config = None
        lines = result.stdout.split("\n")

        for line in lines:
            if kernel_name is None:
                kn_match = kernel_name_pattern.search(line)
                if kn_match:
                    kernel_name = kn_match.group(1)
            match = config_line_pattern.search(line)
            if match:
                chosen_config = {
                    "BLOCK_SIZE_M": int(match.group(1)),
                    "BLOCK_SIZE_N": int(match.group(2)),
                    "BLOCK_SIZE_K": int(match.group(3)),
                    "GROUP_SIZE_M": int(match.group(4)),
                    "num_warps": int(match.group(5)),
                    "num_ctas": int(match.group(6)),
                    "num_stages": int(match.group(7)),
                    "maxnreg": None if match.group(8) == "None" else match.group(8)
                }
        
        if chosen_config is None:
            print(f"Warning: No config found for batch_size={batch_size}, in_features={in_features}, index_size={index_size}")
        else:
            best_configs[key_str] = chosen_config
            print(f"Saved config for batch_size={batch_size}, in_features={in_features}, index_size={index_size}: {chosen_config}")

    if kernel_name is None:
        kernel_name = "unknown_kernel"

    # Update output_path to reflect discovered kernel_name if necessary.
    output_path = output_dir / f"best_configs_{kernel_name}_{in_features}.json"
    with open(output_path, "w") as f:
        json.dump(best_configs, f, indent=4)

    print(f"Best configs saved to {output_path}")


if __name__ == "__main__":
    # add in_features as an argument
    args = arg_parser()
    in_features = args.in_features  # Adjust as needed
    mode = args.mode
    
    if mode != "auto":
        print(f"Running autotuning for in_features={in_features}, mode = {mode}")
        run_offline_autotune_and_record(in_features, mode=mode)
    else:
        print(f"Running autotuning for in_features={in_features}, mode = row")
        run_offline_autotune_and_record(in_features, mode="row")
        
        print(f"Running autotuning for in_features={in_features}, mode = col")
        run_offline_autotune_and_record(in_features, mode="col")