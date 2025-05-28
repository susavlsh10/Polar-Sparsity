import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import os
from HybridTensor.routers.mlp.trainer_mlp import train
from HybridTensor.routers.router_utils import DATA, MODEL_CHOICES, DATA_CHOICES, CONFIG, BasicDataset
from HybridTensor.routers.router_utils import get_data, create_dataset, create_log_path, augment_data, get_data_
from HybridTensor.routers.router_utils import generate_experiment_id, create_date_directory, generate_log_filename, setup_logging, generate_model_filename
import logging
from HybridTensor.routers.mlp.MLP_router import Router

from HybridTensor.utils.utils import _get_device, extract_model_name
from HybridTensor.utils.activations import OPT_CONFIGS, OPT_MODELS

def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch OPT Full Model")
    parser.add_argument("--model", type=str, default="6_7b", choices=MODEL_CHOICES)
    parser.add_argument("--model_name", type=str, default="opt", help="model name")
    parser.add_argument("--model_index", type=int, default=8, help="model index")
    parser.add_argument("--dataset", type=str, default="c4", choices=DATA_CHOICES)
    parser.add_argument("--L", type=int, default=0, help="which layer")
    parser.add_argument("--D", type=int, default=1024, help="low rank dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")  # a smaller batch size results in better precision and recall of the model
    parser.add_argument("--norm", type=str, default="none", choices=["batch", "layer", "none"])
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="<PATH_TO_CHECKPOINT_DIR>", help="checkpoint directory")  # add a argument for checkpoint dir
    parser.add_argument("--data_dir", type=str, default="<PATH_TO_DATA_DIR>", help="data directory")  # add a argument for data dir
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")  # add a argument for which gpu to use
    parser.add_argument("--max_samples", type=int, default=0, help="total samples to process")  # add a argument for total samples
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data_augmentation", type=int, default=0)    # use cold neurons for routing 
    parser.add_argument("--loss_fn", type=str, default="bce", choices=["bce", "focal", "mse"])
    parser.add_argument("--hidden_activation", type=str, default="none", choices=["relu", "gelu", "relu_squared", "leaky_relu", "swish", "swiglu", "elu", "selu", "mish", "none"])
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = arg_parser()
    model_name = OPT_MODELS[args.model_index -1]
    model_config = OPT_CONFIGS[model_name]
    model_dim = model_config['d']
    inner_dim = args.D
    device = _get_device(args.gpu)
    
    # Generate a unique experiment ID, date directory, and log file path
    if args.debug:
        experiment_id = generate_experiment_id()    
        date_dir = create_date_directory(args.ckpt_dir, args.model)  # This will create {base_dir}/{YYYY-MM-DD}/
        log_file_path = generate_log_filename(experiment_id, args.model_name, args.L, date_dir)
    else:
        date_dir = "None"
        experiment_id = "None"
        model_name_clean = extract_model_name(model_name)
        log_file_path = f"{args.ckpt_dir}/{model_name_clean}-routers/mlp/logs/{args.model_name}-{args.L}.log"
        ckpt_path = f"{args.ckpt_dir}/{model_name_clean}-routers/mlp"
    
    # Ensure the logs directory exists
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logging
    setup_logging(log_file_path)
    
    # Log the experiment ID and setup details
    logging.info(f"Experiment ID: {experiment_id}")
    logging.info(f"Model Name: {args.model_name}")
    logging.info(f"Layer Index: {args.L}")
    logging.info(f"Checkpoint Directory: {args.ckpt_dir}")
    logging.info(f"Date Directory: {date_dir}")
    logging.info(f"Using Dropout: {args.dropout}")
    logging.info(f"loss_fn: {args.loss_fn}")
    logging.info(f"hidden_activation: {args.hidden_activation}")
    logging.info(f"Learning Rate: {args.lr}")
    
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 40, "Layer", args.L, "=" * 40)
    # query, labels = get_data(args, args.L) # Deja vu style data loading 
    if args.max_samples == 0:
        total_samples = 400000
    else:
        total_samples = args.max_samples
    
    query, labels = get_data_(args.data_dir, args.L, data_type = 'mlp_activations', total_samples=total_samples) # New data loading style
    
    
    if args.data_augmentation:  # hasn't been helpful yet
        logging.info("Using Data Augmentation")
        index_metadata, cold_labels = augment_data(labels, device=device)
        hot_index, cold_index, pruned_index = index_metadata    # need to store this meta data to storage
        cold_neurons = len(cold_index)
        train_loader, test_loader = create_dataset(query, cold_labels, args)
    else:
        logging.info("Not Using Data Augmentation")
        train_loader, test_loader = create_dataset(query, labels, args)
        cold_neurons = 0
    
    # Initialize the model
    # router = Router(args, cold_neurons=cold_neurons).to(device)
    mlp_router = Router(model_dim, inner_dim, norm=args.norm, hidden_activation=args.hidden_activation)
    
    # Log the model architecture
    logging.info("Model Architecture:\n{}".format(mlp_router))
    logging.info("Training Arguments:\n{}".format(args))
    
    best_model, eval_result = train(
        mlp_router,  train_loader, test_loader, args, device, verbal=True
    )
    # Generate model filename based on evaluation results
    if args.debug:
        model_filename = generate_model_filename(experiment_id, args.model_name, args.L, eval_result)
        model_file_path = date_dir / model_filename
    else:
        model_file_path = f"{ckpt_path}/mlp_router_{args.L}-{eval_result['Recall']:.2f}-{eval_result['Precision']:.2f}-{eval_result['Classifier Sparsity']:.2f}.pt"
    
    # Save the best model
    torch.save(best_model, model_file_path)
    logging.info(f"Model saved at {model_file_path}")
