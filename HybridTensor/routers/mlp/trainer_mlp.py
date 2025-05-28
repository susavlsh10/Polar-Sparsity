import torch
import copy
import numpy as np
from tqdm import tqdm
import logging

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class LossFunction(torch.nn.Module):
    def __init__(self, args, device='cuda:0'):
        super(LossFunction, self).__init__()
        self.loss_fn_str = args.loss_fn
        if args.loss_fn == "bce":
            self.loss_fn = torch.nn.functional.binary_cross_entropy
        elif args.loss_fn == "focal":
            self.loss_fn = FocalLoss(alpha=args.alpha, gamma=args.gamma)
        elif args.loss_fn == "mse":
            self.loss_fn = torch.nn.functional.mse_loss

    def forward(self, logits, targets):
        if self.loss_fn_str == "bce":
            probs = logits.sigmoid()
            weight = (targets.sum() / targets.numel()) + 0.005
            loss_weight = targets * (1 - weight) + weight
            loss = self.loss_fn(
                probs, targets, weight=loss_weight
            )
            return loss
        else:
            return self.loss_fn(logits, targets)

def eval_print(validation_results):
    result = ""
    for metric_name, metirc_val in validation_results.items():
        result = f"{result}{metric_name}: {metirc_val:.4f} "
    return result


def generate_label(y):
    # positive
    one_hot = (y > 0).to(y.dtype)
    return one_hot


def evaluate_better(model, device, loader, args, smalltest=False):
    model.eval()

    eval = {
        "Loss": [],
        "Loss Weight": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }

    total_TP = 0  # True Positives
    total_FP = 0  # False Positives
    total_FN = 0  # False Negatives
    total_TN = 0  # True Negatives
    
    th = args.threshold

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            if args.loss_fn != "mse":
                y = generate_label(y)

            logits = model(x)
            if args.loss_fn == "mse":
                preds = (logits > th).float()
            else:
                probs = logits.sigmoid()
                preds = (probs >= th).float()

            if args.loss_fn == "mse":
                loss = torch.nn.functional.mse_loss(preds, y)
            else:
                # Compute loss with weighting
                weight = (y.sum() / y.numel()) + 0.005
                loss_weight = y * (1 - weight) + weight
                eval["Loss Weight"].append(weight.item())
                loss = torch.nn.functional.binary_cross_entropy(
                    probs, y, weight=loss_weight
                )
            eval["Loss"].append(loss.item())

            # Compute sparsity metrics
            eval["True Sparsity"].append(y.sum(dim=1).float().mean().item())
            eval["Classifier Sparsity"].append(preds.sum(dim=1).float().mean().item())

            # Flatten predictions and labels for metric computation
            preds_flat = preds.view(-1)
            y_flat = y.view(-1)

            # Calculate True Positives, False Positives, False Negatives, True Negatives
            TP = ((preds_flat == 1) & (y_flat == 1)).sum().item()
            FP = ((preds_flat == 1) & (y_flat == 0)).sum().item()
            FN = ((preds_flat == 0) & (y_flat == 1)).sum().item()
            TN = ((preds_flat == 0) & (y_flat == 0)).sum().item()

            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

            if batch_idx >= 100 and smalltest:
                break

    # Average loss and sparsity metrics over all batches
    for k in ["Loss", "Loss Weight", "Classifier Sparsity", "True Sparsity"]:
        eval[k] = np.mean(eval[k])

    # Compute Precision, Recall, and F1 Score
    if total_TP + total_FP > 0:
        precision = total_TP / (total_TP + total_FP)
    else:
        precision = 0.0

    if total_TP + total_FN > 0:
        recall = total_TP / (total_TP + total_FN)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    eval["Precision"] = precision
    eval["Recall"] = recall
    eval["F1 Score"] = f1_score

    return eval

def evaluate_thresholds(model, device, loader, args):
    thresholds = [0.5, 0.4, 0.3, 0.2]
    for th in thresholds:
        eval_results = evaluate_better(model, device, loader, args, smalltest=True)
        print(f"[Threshold {th}] {eval_print(eval_results)}")


def train(model, train_loader, valid_loader, args, device, verbal=True):
    num_val = 0
    early_stop_waiting = 5
    val_inter = len(train_loader) // (num_val + 1) + 1
    # num_print = 0
    # print_inter = len(train_loader) // (num_print + 1) + 1
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01) # 0.01
    eval_results = evaluate_better(model, device, valid_loader, args, smalltest=True)
    
    if verbal:
        # print(f"[Start] {eval_print(eval_results)}")
        logging.info(f"[Start] {eval_print(eval_results)}")

    best_model = copy.deepcopy(model.state_dict())
    base_acc = eval_results["F1 Score"]
    best_eval = eval_results
    no_improve = 0
    
    # Instantiate the loss function
    loss_fn = LossFunction(args, device)
    logging.info(f"Loss Function: {loss_fn}")
    logging.info("Training started.")
    for e in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            model.train()
            x, y = batch
            optimizer.zero_grad()

            y = y.float().to(device)
            if args.loss_fn != "mse":
                y = generate_label(y)
            
            logits = model(x.to(device))
            loss = loss_fn(logits, y)
            
            loss.backward()
            optimizer.step()

            if ((batch_idx + 1) % val_inter == 0 and batch_idx != 0 and batch_idx != len(train_loader)):
                valid_eval_results = evaluate_better(model, device, valid_loader, smalltest=False)
                model.train()
                logging.info(f"[{e}, {batch_idx}] Validation: {eval_print(valid_eval_results)}")

        # evalute on validation set and save the best model
        if (e+1) % 1 == 0:
            epoch_eval_results = evaluate_better(model, device, valid_loader, args, smalltest=False)
            if verbal:
                logging.info(f"[Epoch {e+1}] [Valid] {eval_print(epoch_eval_results)}")

            if epoch_eval_results["F1 Score"] > base_acc:
                base_acc = epoch_eval_results["F1 Score"]
                best_eval = epoch_eval_results
                model.cpu()
                best_model = copy.deepcopy(model.state_dict())
                model.to(device)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= early_stop_waiting or base_acc > 0.99:
                break
            
    # After training, log the best evaluation results
    logging.info("Training completed.")
    logging.info(f"Best Evaluation Results: {eval_print(best_eval)}")
    return best_model, best_eval
