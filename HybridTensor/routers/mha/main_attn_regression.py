import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from HybridTensor.routers.mha.trainer_att import train
from HybridTensor.routers.router_utils import get_data_, create_dataset
from HybridTensor.utils.activations import CONFIGS, MODELS
from HybridTensor.utils.utils import _get_device, extract_model_name
import os


import torch
from HybridTensor.routers.mha.trainer_regression import train_regression

class MHA_Router(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, norm="layer", hidden_activation="none"):
        super(MHA_Router, self).__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        # self.top_k = top_k
        self.norm = norm
        self.hidden_activation = hidden_activation
        
        self.linear1 = torch.nn.Linear(model_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, num_heads)
        
        if hidden_activation != "none":
            self.act = self.activation_fn(hidden_activation)
        if norm == "layer":
            self.norm1 = torch.nn.LayerNorm(inner_dim)
        
    def activation_fn(self, hidden_activation):
        if hidden_activation == "relu":
            return torch.nn.ReLU()
        elif hidden_activation == 'gelu':
            return torch.nn.GELU()
        else:
            # defaults to relu
            return torch.nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        if self.hidden_activation != "none":
            out = self.act(out)
        
        if self.norm == "layer":
            out = self.norm1(out)
        
        out = self.linear2(out)
        
        return out
    
class MHA_Router_Linear(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads):
        super(MHA_Router_Linear, self).__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        
        self.linear1 = torch.nn.Linear(model_dim, num_heads)
        
    def forward(self, x):
        out = self.linear1(x)
        return out



def arg_parser():
    parser = argparse.ArgumentParser(description=" MHA Router Training")
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b", choices=MODELS)
    parser.add_argument("--model_name", type=str, default="opt", help="model name")
    parser.add_argument("--model_index", type=int, default=8, help="model index")
    parser.add_argument("--L", type=int, default=4, help="which layer")
    parser.add_argument("--D", type=int, default=1024, help="low rank dimension")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")  # a smaller batch size results in better precision and recall of the model
    parser.add_argument("--epochs", type=int, default=5, help="epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--k", type=float, default=0.4, help="top k percent to mark as activate head")
    parser.add_argument("--hidden_activation", type=str, default="none", choices=["none", "relu", "gelu"])
    parser.add_argument("--data_dir", type=str, default="/home/grads/s/sls7161/nvme/HybridTensor/opt-6.7b_act_data/", help="data directory")  # add a argument for data dir
    parser.add_argument("--ckpt_dir", type=str, default="/home/grads/s/sls7161/nvme/HybridTensor/checkpoint", help="checkpoint directory")  # add a argument for checkpoint dir
    parser.add_argument("--total_samples", type=int, default=0, help="total samples")  # add a argument for total samples
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")  # add a argument for which gpu to use
    parser.add_argument("--norm", type=str, default="none", choices=["layer", "batch", "none"])
    
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    
    # Set random seed for reproducibility
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # load the attention activation data
    if args.total_samples == 0:
        args.total_samples = 400000
    else:
        args.total_samples = args.total_samples
    
    hidden_states, attn_norms = get_data_(args.data_dir, args.L, data_type="attn_norms", total_samples=args.total_samples)
    train_loader, test_loader = create_dataset(hidden_states, attn_norms, args)
    
    model_name = MODELS[args.model_index -1]
    model_config = CONFIGS[model_name]
    model_dim = model_config['d']
    num_heads = model_config['h']
    inner_dim = args.D
    device = _get_device(args.gpu)
    
    print("=" * 40, "Layer", args.L, "=" * 40)

    # mha_router = MHA_Router(model_dim, inner_dim, num_heads, norm=args.norm, hidden_activation=args.hidden_activation)
    mha_router = MHA_Router_Linear(model_dim, inner_dim, num_heads)
    
    # print(mha_router)
    best_model, eval_result = train_regression(mha_router,  train_loader, test_loader, args, device, verbal=True)
    
    # save the checkpoint
    model_name_clean = extract_model_name(model_name)
    ckpt_path = f"{args.ckpt_dir}/{model_name_clean}-routers/attn_regression"
    
    # create the checkpoint directory if it does not exist
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    file_name = f"{ckpt_path}/attn_router_{args.L}-{eval_result['Recall']:.4f}-{eval_result['Precision']:.4f}.pt"
    torch.save(best_model, file_name)
    print(f"Model saved at {file_name}")