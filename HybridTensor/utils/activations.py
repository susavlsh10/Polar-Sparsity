import os
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    "facebook/opt-125m",        # 1
    "facebook/opt-350m",        # 2
    "facebook/opt-1.3b",        # 3
    "facebook/opt-2.7b",        # 4
    "facebook/opt-6.7b",        # 5
    "facebook/opt-13b",         # 6
    "facebook/opt-30b",         # 7
    "facebook/opt-66b",          # 8
    "SparseLLM/ReluLLaMA-7B",   # 9
    "SparseLLM/prosparse-llama-2-7b", # 10
    "meta-llama/Llama-2-7b-hf",     # 11
    "meta-llama/Llama-2-13b-hf",    # 12
    "-",
    "meta-llama/Llama-3.1-8B",      # 14
    "meta-llama/Llama-3.1-70B",     # 15
    
]


CONFIGS = {
    'facebook/opt-175b':{
        'num_layer': 95,
        'd':12288,
        'h': 96,
        'neurons': 49152
    },
    'facebook/opt-66b':{
        'num_layer': 64,
        'd':9216,
        'h': 72,
        'neurons': 36864,
        'layer_imp': "results/attn_importance/opt-66b_attn_importance.json"
    },
    'facebook/opt-30b':{
        'num_layer': 48,
        'd':7168,
        'h': 56,
        'neurons': 28672,
        'layer_imp': "results/attn_importance/opt-30b_attn_importance.json"
    },
    'facebook/opt-13b':{
        'num_layer': 40,
        'd':5120,
        'h': 40,
        'neurons': 20480,
        'layer_imp': "results/attn_importance/opt-13b_attn_importance.json"
    },
    'facebook/opt-6.7b':{
        'num_layer': 32,
        'd':4096,
        'h': 32,
        'neurons': 16384,
        'layer_imp': "results/attn_importance/opt-6.7b_attn_importance.json"
    },
    'facebook/opt-2.7b':{
        'num_layer': 32,
        'd':2560,
        'h': 32,
        'neurons': 10240
    },
    'facebook/opt-1.3b':{
        'num_layer': 24,
        'd':2048,
        'h': 32,
        'neurons': 8192
    },
    'facebook/opt-350m':{
        'num_layer': 24,
        'd':1024,
        'h': 16,
        'neurons': 4096
    },
    'facebook/opt-125m':{
        'num_layer': 12,
        'd':768,
        'h': 12,
        'neurons': 3072
    },
    'SparseLLM/ReluLLaMA-7B':{
        'num_layer': 32,
        'd': 4096,
        'h': 32,
        'neurons': 11008
    },
    'SparseLLM/prosparse-llama-2-7b':{
        'num_layer': 32,
        'd': 4096,
        'h': 32,
        'neurons': 11008
    },
    'meta-llama/Llama-2-7b-hf':{
        'num_layer': 32,
        'd': 4096,
        'h': 32,
        'neurons': 11008,
        'layer_imp': "results/attn_importance/Llama-2-7b-hf_attn_importance.json"
    },
    'meta-llama/Llama-2-13b-hf':{
        'num_layer': 40,
        'd': 5120,
        'h': 40,
        'neurons': 13824,
        'layer_imp': "results/attn_importance/Llama-2-13b-hf_attn_importance.json"
    },
    'meta-llama/Llama-3.1-8B':{
        'num_layer': 32,
        'd': 4096,
        'h': 32,
        'neurons': 14336,
        'layer_imp': "results/attn_importance/Llama-3.1-8B_attn_importance.json"
    },
    'meta-llama/Llama-3.1-70B':{
        'num_layer': 80,
        'd': 8192,
        'h': 64,
        'neurons': 28672,
        'layer_imp': "results/attn_importance/Llama-3.1-70B_attn_importance.json"
    }
    
}


# This class is used to store the activation thresholds for each layer of the model. Used for evaluation purposes.
class ActivationThresholds:
    def __init__(self, num_layers, attn_th=0.0, mlp_th = 0.0):
        self.activation_threshold = {}
        self.mlp_threshold = {}
        for i in range(num_layers):
            self.activation_threshold[i] = attn_th
            self.mlp_threshold[i] = mlp_th

    def set_threshold(self, layer_idx, threshold):
        self.activation_threshold[layer_idx] = threshold

    def get_threshold(self, layer_idx):
        return self.activation_threshold[layer_idx]

    def save_thresholds(self, file_path):
        with open(file_path, 'w') as file:
            documents = yaml.dump(self.activation_threshold, file)

    # def load_thresholds(self, file_path):
    #     with open(file_path, 'r') as file:
    #         self.activation_threshold = yaml.load(file, Loader=yaml.FullLoader)
    
    def load_thresholds(self, sparsity_map):
        """
        Load the activation thresholds from a given sparsity map.
        
        Parameters:
            sparsity_map (dict): A dictionary containing layer indices and their corresponding activation thresholds.
        """
        for layer_idx, threshold in sparsity_map.items():
            self.activation_threshold[layer_idx] = threshold
    
    @classmethod
    def from_file(cls, file_path):
        """Class method to create an instance from a YAML file."""
        with open(file_path, 'r') as file:
            activation_threshold = yaml.load(file, Loader=yaml.FullLoader)
        
        # Create an instance and set its activation_threshold
        instance = cls()
        instance.activation_threshold = activation_threshold
        return instance
    
    
def build_mlp_topk_lookup(data_path: str, batch_size: int, delta: int = 128) -> dict:
    """
    Creates a lookup table from the CSV file 'mlp_act_batch_{batch_size}_stats.csv' in the given directory.
    
    The lookup maps each layer to a top-k value calculated as:
      top_k = ceil((average_activation + std_activation + delta))
    
    Parameters:
        data_path (str): The path to the directory containing the CSV files.
        batch_size (int): The batch size to use in the file name and for filtering.
        delta (float): The additional value added to the activations before rounding.
    
    Returns:
        dict: A mapping from layer id to computed top-k value.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist in the provided directory.
    """
    file_name = f"mlp_act_batch_{batch_size}_stats.csv"
    full_path = os.path.join(data_path, file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(full_path)
    
    def calc_top_k(avg, std, delta):
        raw_value = avg + std + delta
        # return int(np.ceil(raw_value /128) * 128)    # Round up to the nearest multiple of 128
        return int(np.ceil(raw_value))
    
    mlp_lookup = {
        row["layer"]: calc_top_k(row["average_activation"], row["std_activation"], delta)
        for _, row in df.iterrows() if row["batch_size"] == batch_size
    }
    
    return mlp_lookup


def _update_hf_mlp_topk(model, mlp_lookup):
    """
    Updates the top-k values in the model's MLP layers using the provided lookup table.
    
    Parameters:
        model (HybridModel): The model to update.
        mlp_lookup (dict): The lookup table mapping layer id to top-k value.
        delta (int): The additional value added to the activations before rounding.
    """
    for layer_idx, top_k in mlp_lookup.items():
        model.model.decoder.layers[layer_idx].mlp_act= int(top_k)


def identify_model_type(model_name):
    """
    Identifies if the given model name is an OPT model or a Llama model.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: "OPT" if the model is an OPT model, "Llama" if it's a Llama model, or "Unknown".
    """
    if "opt" in model_name.lower():
        return "OPT"
    elif "llama" in model_name.lower():
        return "Llama"
    else:
        return "Unknown"


OPT_MODELS = [
    "facebook/opt-125m",    # 1
    "facebook/opt-350m",    # 2
    "facebook/opt-1.3b",    # 3
    "facebook/opt-2.7b",    # 4
    "facebook/opt-6.7b",    # 5
    "facebook/opt-13b",     # 6
    "facebook/opt-30b",     # 7
    "facebook/opt-66b"      # 8
]


OPT_CONFIGS = {
    'facebook/opt-175b':{
        'num_layer': 95,
        'sp_config': None,
        'd':12288,
        'h': 96,
    },
    'facebook/opt-66b':{
        'num_layer': 64,
        'd':9216,
        'h': 72,
    },
    'facebook/opt-30b':{
        'num_layer': 48,
        'd':7168,
        'h': 56,
    },
    'facebook/opt-13b':{
        'num_layer': 40,
        'd':5120,
        'h': 40,
    },
    'facebook/opt-6.7b':{
        'num_layer': 32,
        'd':4096,
        'h': 32,
    },
    'facebook/opt-2.7b':{
        'num_layer': 32,
        'd':2560,
        'h': 32,
    },
    'facebook/opt-1.3b':{
        'num_layer': 24,
        'd':2048,
        'h': 32,
    },
    'facebook/opt-350m':{
        'num_layer': 24,
        'd':1024,
        'h': 16,
    },
    'facebook/opt-125m':{
        'num_layer': 12,
        'd':768,
        'h': 12,
    },
}

'''
import seaborn as sns
def plot_average_activation(directory_path, model_name):


    # Read all the .csv files in the specified directory and concatenate them into a single DataFrame
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    df = pd.concat([pd.read_csv(f) for f in files])
    
    # Set the seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    total_neurons = OPT_CONFIGS[model_name]["d"] * 4
    # Compute the average activation percentage and standard deviation percentage
    df['average_activation_percentage'] = (df['average_activation'] / total_neurons) * 100
    df['std_activation_percentage'] = (df['std_activation'] / total_neurons) * 100

    # Create a color palette with a different color for each batch size
    palette = sns.color_palette("husl", df['batch_size'].nunique())

    # Initialize the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Loop over each batch size and plot the average_activation_percentage with error bars
    for i, (batch_size, group) in enumerate(df.groupby('batch_size')):
        plt.errorbar(
            group['layer'],
            group['average_activation_percentage'],
            yerr=group['std_activation_percentage'],
            label=f'Batch Size {batch_size}',
            capsize=3,
            marker='o',
            linestyle='-',
            color=palette[i]
        )
        
    # Set y-axis ticks at every 10% increment
    plt.yticks(range(0, 101, 10))  # Y-ticks from 0% to 100% in steps of 10%
    
    # Shaded region
    plt.axhspan(80, plt.ylim()[1], facecolor='gray', alpha=0.1)
    
    # Set the labels and title of the plot
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Average Activation Percentage (%)', fontsize=14)
    plt.title(f'Model: {model_name} Average Activation Percentage vs Layer for Different Batch Sizes', fontsize=16)
    
    # Show legend
    plt.legend(title='Batch Size', fontsize=12, title_fontsize=12)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the image
    plt.savefig('average_activation_analysis.png')
    
    # Display the plot
    plt.show()
    


'''
