import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from HybridTensor.routers.router_utils import get_data_, create_dataset
from HybridTensor.routers.mha.trainer_att import train
from HybridTensor.utils.utils import _get_device, extract_model_name
# from HybridTensor.utils.activations import OPT_CONFIGS, OPT_MODELS
from HybridTensor.utils.activations import CONFIGS, MODELS
from HybridTensor.routers.mha.main_attn_regression import MHA_Router, MHA_Router_Linear

import os 


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
    parser.add_argument("--k", type=float, default=0.3, help="top k percent to mark as activate head")  # top k from attention sweep 
    parser.add_argument("--hidden_activation", type=str, default="none", choices=["none", "relu", "gelu"])
    parser.add_argument("--data_dir", type=str, default="<PATH_TO_DATA_DIR>", help="data directory")  # add an argument for data dir
    parser.add_argument("--ckpt_dir", type=str, default="<PATH_TO_CHECKPOINT_DIR>", help="checkpoint directory")  # add an argument for checkpoint dir
    parser.add_argument("--total_samples", type=int, default=0, help="total samples")  # add a argument for total samples
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")  # add a argument for which gpu to use
    parser.add_argument("--norm", type=str, default="none", choices=["layer", "batch", "none"])
    parser.add_argument("--router_arch", type=str, default="linear", choices=["mlp", "linear"])
    
    return parser.parse_args()


def main():

    args = arg_parser()
    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    # load the attention activation data
    if args.total_samples == 0:
        args.total_samples = 400000
    else:
        args.total_samples = args.total_samples
    
    model_name = MODELS[args.model_index -1]
    model_config = CONFIGS[model_name]
    model_dim = model_config['d']
    num_heads = model_config['h']
    inner_dim = args.D
    device = _get_device(args.gpu)

    print("=" * 40, "Layer", args.L, "=" * 40)

    # query, labels = get_data(args, args.L)
    hidden_states, attn_norms = get_data_(args.data_dir, args.L, data_type="attn_norms", total_samples=args.total_samples)
    train_loader, test_loader = create_dataset(hidden_states, attn_norms, args)

    if args.router_arch == "mlp":
        mha_router = MHA_Router(model_dim, inner_dim, num_heads, norm=args.norm, hidden_activation=args.hidden_activation)
    else:   
        mha_router = MHA_Router_Linear(model_dim, inner_dim, num_heads)
    
    print(mha_router)
    print("Start Training")
    best_model, eval_result = train(
        mha_router,  train_loader, test_loader, args, device, verbal=True
    )

    # save the checkpoint
    model_name_clean = extract_model_name(model_name)
    ckpt_path = f"{args.ckpt_dir}/{model_name_clean}-routers/attn_classifier"
    
    # create the checkpoint directory if it does not exist
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    file_name = f"{ckpt_path}/attn_router_{args.L}-{eval_result['Recall']:.4f}-{eval_result['Precision']:.4f}.pt"
    torch.save(best_model, file_name)
    print(f"Model saved at {file_name}")


if __name__ == "__main__":
    main()
