import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, auc, roc_curve

def evaluate_metrics(predictions: np.ndarray,
                     binary_preds: np.ndarray,
                     targets: np.ndarray,
                     use_logits: bool = False,
                     metrics: list = ['AUROC'],
                     evaluation_sub_tasks: list = None,
                     tasks:list=None) -> dict:
    """
    Compute evaluation metrics for model or ensemble outputs.
    Args:
        predictions (np.ndarray): Array of shape (N, C) with raw logits (if use_logits=True) or probabilities.
        binary_preds (np.ndarray): Array of shape (N, C) containing final binary predictions (0/1).
        targets (np.ndarray): Array of shape (N, C) with ground-truth binary labels.
        use_logits (bool): Whether `predictions` are raw logits (will apply sigmoid internally).
        metrics (list): Metrics to compute: any of 'AUROC', 'F1', 'Youden'.
        average_auroc_classes (list): Optional list of class indices over which to average AUROC.
    Returns:
        dict: Contains per-class scores and any requested summary statistics.
    """
    results = {}
    # Convert logits to probabilities if needed    
    probs = predictions
    if use_logits:
        probs = 1 / (1 + np.exp(-predictions))

    # AUROC per class
    if 'AUROC' in metrics:
        auroc_map = {}
        for task_idx in range(len(tasks)):
            y_true = targets[:, task_idx]
            y_score = probs[:, task_idx]
            if len(np.unique(y_true)) == 2:
                val = roc_auc_score(y_true, y_score)
            else:
                val = np.nan
            key = tasks[task_idx]
            auroc_map[key] = val
        results['AUROC'] = auroc_map
        # Average AUROC over specified subset
        if evaluation_sub_tasks:
            subset = []
            for task in evaluation_sub_tasks:
                val = auroc_map.get(task)
                subset.append(val)
            results['AUROC_subset_mean'] = float(np.mean(subset))

    # F1-score per class - best value at 1 and worst score at 0
    if 'F1' in metrics:
        f1_scores_map = {}
        for task_idx, task in enumerate(tasks):
            score = f1_score(targets[:, task_idx], binary_preds[:, task_idx], zero_division=0.0)
            key = tasks[task_idx]
            f1_scores_map[key] = score
        results['F1_per_class'] = f1_scores_map
        # Average F1 score over specified subset
        if evaluation_sub_tasks:
            subset = []
            for task in evaluation_sub_tasks:
                val = f1_scores_map.get(task)
                subset.append(val)
            results['F1_subset_mean'] = float(np.mean(subset))

    # Youden index per class
    # Best value at 1 (perfect trade-off between sensitivity and specificity) and worst score at 0 (random)
    if 'Youden' in metrics:
        youden_indices_map = {}
        for task_idx, task in enumerate(tasks):
            y_true = targets[:, task_idx]
            y_pred = binary_preds[:, task_idx]
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            youden_idx = (sensitivity + specificity - 1)
            key = tasks[task_idx]
            youden_indices_map[key] = youden_idx
        results['Youden_per_class'] = youden_indices_map
        # Average Youden-index over specified subset
        if evaluation_sub_tasks:
            subset = []
            for task in evaluation_sub_tasks:
                val = youden_indices_map.get(task)
                subset.append(val)
            results['Youden-index_subset_mean'] = float(np.mean(subset))
        
    return results


def find_optimal_thresholds(probabilities: np.ndarray,
                            ground_truth: np.ndarray,
                            tasks: list,
                            metric: str = 'f1',
                            step: float = 0.01) -> (dict, dict):
    """
    Find per-class optimal thresholds to maximize a given metric.
    Args:
        probabilities (np.ndarray): Shape (N, C) of predicted probabilities.
        ground_truth (np.ndarray): Shape (N, C) of true binary labels.
        tasks (list): Class names (for reference; not used in computation).
        metric (str): Metric to optimize: 'f1' or 'youden'.
        step (float): Granularity of threshold search (e.g., 0.01).
    Returns:
        optimal_thresholds (dict): Mapping class index -> best threshold.
        metric_scores (dict): Mapping class index -> best metric value at threshold.
    """
    n_classes = probabilities.shape[1]
    thresholds = np.arange(0.0, 1.0 + step, step)
    optimal_thresholds = {}
    metric_scores = {}

    for task_idx in range(n_classes):
        best_thresh = 0.5
        best_score = -np.inf
        y_probs = probabilities[:, task_idx]
        y_true = ground_truth[:, task_idx]

        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            optimal_thresholds[task_idx] = best_thresh
            metric_scores[task_idx] = None
            continue

        # Search for the threshold that maximizes the chosen metric
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            if metric.lower() == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric.lower() == 'youden':
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sensitivity + specificity - 1
            else:
                raise ValueError("Metric must be 'f1' or 'youden'.")

            if score > best_score:
                best_score = score
                best_thresh = t

        optimal_thresholds[task_idx] = best_thresh
        metric_scores[task_idx] = best_score

    return optimal_thresholds, metric_scores


def threshold_based_predictions(probs: torch.Tensor,
                                thresholds: dict,
                                tasks: list) -> torch.Tensor:
    """
    Apply class-wise thresholds to probabilities to generate binary outputs.
    Args:
        probs (torch.Tensor): Shape (N, C) of probabilities.
        thresholds (dict): Class-index -> threshold value.
        tasks (list): Class names (for interface consistency).
    Returns:
        torch.Tensor: Shape (N, C) of binary predictions (0.0 or 1.0).
    """
    preds = torch.zeros_like(probs)
    for c, thresh in thresholds.items():
        preds[:, c] = (probs[:, c] >= thresh).float()
    return preds


def plot_roc(predictions:np.ndarray=None, 
             ground_truth:np.ndarray=None, 
             tasks:list=None, 
             save_dir:str='results/plots'):
    """
    Computes per-class AUROC using continuous probabilities and saves individual
    and combined ROC plots to the specified directory.

    Args:
        predictions (array-like): shape (n_samples,) or (n_samples, n_classes)
        ground_truth (array-like): shape (n_samples,) or (n_samples, n_classes)
        tasks (list of str): Names for each class/task
        save_dir (str): Path to directory where ROC plots should be saved
        n_classes (int): must match the number of classes. Default is 14
    """
    # Convert to numpy arrays if necessary
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Combined ROC figure
    fig_combined, ax_combined = plt.subplots(figsize=(8, 6))

    for i, task in enumerate(tasks):
        gt = ground_truth[:, i]
        pred = predictions[:, i]

        # Skip if only one class present in ground truth
        if len(np.unique(gt)) < 2:
            print(f"Warning: Only one sample present in ground truth for {task}. Skip.")
            continue

        # Compute ROC
        fpr, tpr, _ = roc_curve(gt, pred)
        roc_auc = auc(fpr, tpr)
        #print(f"Plot ROC for {task}, AUC={roc_auc:.2f}")

        # Individual ROC plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], '--', lw=2, color='gray')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for {task}')
        ax.legend(loc="lower right")

        # Save individual plot
        individual_path = os.path.join(save_dir, f'roc_{task}.png')
        fig.savefig(individual_path)
        plt.close(fig)

        # Add to combined
        ax_combined.plot(fpr, tpr, lw=0.8, label=f'{task} (AUC = {roc_auc:.2f})')

    # Finalize combined plot
    ax_combined.plot([0, 1], [0, 1], '--', lw=2, color='gray')
    ax_combined.set_xlim([0, 1])
    ax_combined.set_ylim([0, 1.05])
    ax_combined.set_xlabel('False Positive Rate')
    ax_combined.set_ylabel('True Positive Rate (sensitivity)')
    ax_combined.set_title('Combined ROC Curves for All Tasks')
    ax_combined.legend(loc="lower right", fontsize='small')

    combined_path = os.path.join(save_dir, 'combined_roc.png')
    fig_combined.savefig(combined_path)
    plt.close(fig_combined)