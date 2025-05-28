
import torch
import copy
import numpy as np
from tqdm import tqdm

# from torch.cuda.amp import autocast, GradScaler
from torch.amp import GradScaler, autocast

def eval_print(validation_results):
    result = ""
    for metric_name, metric_val in validation_results.items():
        result = f"{result}{metric_name}: {metric_val:.4f} "
    return result

def generate_topk_labels(norm, k):
    """
    Generate multi-hot labels by selecting top k indices per sample.
    """
    K = k
    if K <= 0:
        raise ValueError("k must be a positive integer.")
    # Handle cases where k > num_heads
    K = min(K, norm.shape[1])
    _, indices = norm.topk(K, dim=1)
    one_hot = torch.zeros_like(norm).scatter_(1, indices, 1)
    return one_hot

def evaluate_regression(model, device, loader, args, smalltest=False):
    """
    Evaluate the model using regression loss and classification metrics based on top k predictions.
    """
    model.eval()

    eval_metrics = {
        "Loss": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }
    mse_loss = torch.nn.MSELoss()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, y = batch  # x: [batch_size, model_dim], y: [batch_size, num_heads]
            x = x.to(device)
            y = y.float().to(device)

            logits = model(x)
            preds = logits  # [batch_size, num_heads]

            loss = mse_loss(preds, y)
            eval_metrics["Loss"].append(loss.item())

            # Generate top k labels for true and predictions
            K = int(args.k * y.shape[1])
            K_pred = int((args.k+0.1) * y.shape[1])
            topk_true = generate_topk_labels(y, K)
            topk_pred = generate_topk_labels(preds, K_pred)

            # Convert to integer for metric calculations
            topk_true = topk_true.int()
            topk_pred = topk_pred.int()

            # Compute precision, recall, and F1 per sample
            TP = (topk_pred & topk_true).sum(dim=1).float()
            FP = (topk_pred & (1 - topk_true)).sum(dim=1).float()
            FN = ((1 - topk_pred) & topk_true).sum(dim=1).float()

            per_sample_precision = torch.where(
                (TP + FP) > 0,
                TP / (TP + FP),
                torch.zeros_like(TP)
            )
            per_sample_recall = torch.where(
                (TP + FN) > 0,
                TP / (TP + FN),
                torch.zeros_like(TP)
            )
            per_sample_f1 = torch.where(
                (per_sample_precision + per_sample_recall) > 0,
                2 * per_sample_precision * per_sample_recall / (per_sample_precision + per_sample_recall),
                torch.zeros_like(per_sample_precision)
            )

            eval_metrics["Precision"].append(per_sample_precision.mean().item())
            eval_metrics["Recall"].append(per_sample_recall.mean().item())
            eval_metrics["F1 Score"].append(per_sample_f1.mean().item())

            # Sparsity metrics
            # eval_metrics["True Sparsity"].append(topk_true.sum(dim=1).mean().item())
            # eval_metrics["Classifier Sparsity"].append(topk_pred.sum(dim=1).mean().item())
            
            eval_metrics["True Sparsity"].append(topk_true.sum(dim=1).float().mean().item())
            eval_metrics["Classifier Sparsity"].append(topk_pred.sum(dim=1).float().mean().item())


            if smalltest and batch_idx >= 100:
                break

    # Average metrics over all batches
    for k, v in eval_metrics.items():
        eval_metrics[k] = np.mean(v)

    return eval_metrics

def train_regression(model, train_loader, valid_loader, args, device, verbal=True):
    """
    Train the MHA_Router model as a regression model to predict attention norms.
    """
    early_stop_waiting = 2
    scaler = GradScaler()
    # scaler = GradScaler(device_type='cuda')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    mse_loss_fn = torch.nn.MSELoss()

    # Initial evaluation
    eval_results = evaluate_regression(model, device, valid_loader, args, smalltest=True)
    if verbal:
        print(f"[Start] {eval_print(eval_results)}")

    best_model = copy.deepcopy(model.state_dict())
    base_recall = eval_results["Recall"]
    best_eval = eval_results
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            x, y = batch
            x = x.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                preds = model(x)  # [batch_size, num_heads]
                loss = mse_loss_fn(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            if verbal and (batch_idx + 1) % 100 == 0:
                # print(f"[Epoch {epoch+1}, Batch {batch_idx+1}] Loss: {loss.item():.4f}")
                pass

        # End of epoch evaluation
        train_eval_results = evaluate_regression(model, device, train_loader, args, smalltest=True)
        valid_eval_results = evaluate_regression(model, device, valid_loader, args, smalltest=False)

        if verbal:
            print(f"[Epoch {epoch+1}] [Train] {eval_print(train_eval_results)}")
            print(f"[Epoch {epoch+1}] [Valid] {eval_print(valid_eval_results)}\n")

        # Check for improvement
        if valid_eval_results["Recall"] > base_recall:
            base_recall = valid_eval_results["Recall"]
            best_eval = valid_eval_results
            best_model = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= early_stop_waiting or base_recall > 0.99:
            if verbal:
                print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    return best_model, best_eval