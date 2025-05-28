import json
import bisect
import os
import glob
import ast

class HeuristicSelector:
    def __init__(self, config_dir, type = 'row'):
        """
        Initialize the HeuristicSelector by loading all configuration files in the specified directory.

        Args:
            config_dir (str): Path to the directory containing configuration JSON files.
        """
        self.configs = {}  # in_features -> (M, K, N) -> sorted list of (index_size, config)
        self.type = type
        self._load_configs(config_dir)
        

    def _load_configs(self, config_dir):
        """
        Load all JSON configuration files from the specified directory.

        Args:
            config_dir (str): Path to the directory containing configuration JSON files.
        """
        # Define the pattern to match configuration files
        # pattern = os.path.join(config_dir, 'best_configs_matmul_gather_kernel_col_*.json')
        pattern = os.path.join(config_dir, f'best_configs_matmul_gather_kernel_{self.type}_*.json')
        config_files = glob.glob(pattern)

        if not config_files:
            # raise ValueError(f"No configuration files found in directory: {config_dir}")
            print(f"No configuration files found in directory: {config_dir}. Precompile the kernels for faster inference.")
            return

        for file_path in config_files:
            # Extract in_features from the file name
            basename = os.path.basename(file_path)
            try:
                # Assuming the file name pattern includes in_features as an integer
                in_features_str = basename.split('_')[-1].replace('.json', '')
                in_features = int(in_features_str)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Filename {basename} does not match the expected pattern.") from e

            with open(file_path, 'r') as f:
                saved_configs = json.load(f)

            # Convert string keys back to tuples and organize by (M, K, N)
            temp_dict = {}
            for key_str, cfg in saved_configs.items():
                try:
                    # Safely parse the tuple string
                    M, K, N, idx_size = ast.literal_eval(key_str)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid key format: {key_str}") from e

                temp_dict.setdefault((M, K, N), []).append((idx_size, cfg))

            # Sort the index_size for each (M, K, N)
            for mkn, entries in temp_dict.items():
                entries.sort(key=lambda x: x[0])  # Sort by index_size

                # Initialize the in_features dictionary if not present
                if in_features not in self.configs:
                    self.configs[in_features] = {}

                self.configs[in_features][mkn] = entries

    def get_config(self, M, K, N, index_size):
        """
        Retrieve the best configuration based on the provided parameters.

        Args:
            in_features (int): The input features parameter.
            M (int): Parameter M. Batch size
            K (int): Parameter K. in_features
            N (int): Parameter N. total neurons or hidden_features
            index_size (int): The desired index size.

        Returns:
            dict: The best configuration dictionary.

        Raises:
            ValueError: If no configurations are found for the given parameters.
        """
        in_features = K
        N = in_features * 4
            
        # Check if the specified in_features is available
        if in_features not in self.configs:
            raise ValueError(f"No configurations found for in_features={in_features}")

        mkn = (M, K, N)
        if mkn not in self.configs[in_features]:
            raise ValueError(f"No configurations found for (M={M}, K={K}, N={N}) with in_features={in_features}")

        entries = self.configs[in_features][mkn]
        # entries is sorted by index_size. Find the smallest index_size >= requested index_size.
        index_sizes = [e[0] for e in entries]
        pos = bisect.bisect_left(index_sizes, index_size)

        if pos < len(entries):
            # Found an index_size >= requested index_size
            return entries[pos][1]
        else:
            # No larger index_size found. Fallback to the largest available index_size.
            return entries[-1][1]