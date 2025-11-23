import numpy as np
import pandas as pd
import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, auc, roc_curve
import umap # type: ignore

import plotly.express as px # type: ignore

def evaluate_metrics(predictions: np.ndarray,
                     binary_preds: np.ndarray,
                     targets: np.ndarray,
                     use_logits: bool = False,
                     metrics: list = ['AUROC'],
                     evaluation_sub_tasks: list = None,
                     tasks:list=None) -> dict:
    """
    Compute per-class and subset metrics for model/ensemble outputs.

    Parameters
    ----------
    predictions : (N, C) np.ndarray Raw logits (if use_logits=True) or probabilities [0,1].
    binary_preds : (N, C) np.ndarray Final 0/1 predictions after thresholding.
    targets : (N, C) np.ndarray Ground-truth binary labels.
    use_logits : bool If True, apply sigmoid to `predictions` before AUROC.
    metrics : list[str] Any of {'AUROC','F1','Youden','Accuracy'}; order is ignored.
    evaluation_sub_tasks : list[str] | None Optional subset of class names for averaged summaries (_subset_mean).
    tasks : list[str] All class names; defines column order in arrays.

    Returns
    -------
    dict
        Mapping with keys for each requested metric (per-class dicts) and
        optional subset means (e.g. 'F1_subset_mean').
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

    if 'Accuracy' in metrics:
        accuracy_map = {}
        for task_idx, task in enumerate(tasks):
            y_true = targets[:, task_idx]
            y_pred = binary_preds[:, task_idx] 

            correct = (y_pred == y_true).astype(float).sum()
            total = targets.shape[0]
            accuracy = correct / total

            key = tasks[task_idx]
            accuracy_map[key] = accuracy
        results['Accuracy_per_class'] = accuracy_map
        # Accuracy over specified subset
        if evaluation_sub_tasks:
            subset = []
            for task in evaluation_sub_tasks:
                val = accuracy_map.get(task)
                subset.append(val)
            results['Accuracy_subset_mean'] = float(np.mean(subset))
     
    return results


def find_optimal_thresholds(probabilities, ground_truth, tasks, step=0.01, metric="f1"):
    """
    Search of per-class thresholds to maximize a metric.

    Parameters
    ----------
    probabilities : (N, C) np.ndarray Class probabilities.
    ground_truth : (N, C) np.ndarray Binary labels.
    tasks : list[str] Class names in column order.
    step : float Threshold increment in [0,1]; total ~1/step trials per class.
    metric : {"f1","youden"} Objective to maximize.

    Returns
    -------
    optimal_thresholds : dict[str, float] Best threshold per class name.
    metric_score_dict : dict[str, float] Achieved metric value at the best threshold per class.

    Notes
    -----
    - For large N, consider a quantile/grid reduction to speed up.
    """
    optimal_thresholds = {}
    metric_score_dict = {}
    thresholds = np.arange(0, 1 + step, step)
    #n_classes = probabilities.shape[1]
    for i, task in enumerate(tasks):
        best_metric_score = -float("inf")
        best_threshold = 0.5  # default if nothing better is found
        for t in thresholds:
            preds = (probabilities[:, i] >= t).astype(int) # Bolean value (0,1) will become integer

            if metric == "f1":
                # F1-score = harmonic_mean(precision, sensitivity)=2TP/(2TP+FP+FN))
                score = f1_score(ground_truth[:, i], preds, zero_division=0.0) # If current score better than others -> overwrite
            elif metric == "youden":
                # Compute confusion matrix components
                tp = ((ground_truth[:, i] == 1) & (preds == 1)).sum()
                fn = ((ground_truth[:, i] == 1) & (preds == 0)).sum()
                tn = ((ground_truth[:, i] == 0) & (preds == 0)).sum()
                fp = ((ground_truth[:, i] == 0) & (preds == 1)).sum()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sensitivity + specificity - 1
            else:
                raise ValueError("Invalid metric selected. Must be either 'f1' or 'youden'.")
            
            if score > best_metric_score:
                best_metric_score = score
                best_threshold = t
        optimal_thresholds[task] = best_threshold
        metric_score_dict[task] = best_metric_score
    return optimal_thresholds, metric_score_dict



def threshold_based_predictions(probs: torch.Tensor,
                                thresholds,
                                tasks: list) -> torch.Tensor:
    """
    Apply class‑wise thresholds to probabilities to generate binary outputs.
    Supports both:
      - probs.shape == (N, C): multi‑class
      - probs.shape == (N,):   single‑class (already selected)
    Args:
        probs (torch.Tensor): Shape (N, C) or (N,) of probabilities.
        thresholds (dict or np.ndarray): per‑class thresholds.
        tasks (list): Class names (not used here, kept for interface).
    Returns:
        torch.Tensor: Shape matches input `probs`, with 0.0/1.0 predictions.
    """
    # ---- 1) single‑class case: probs dim == 1 ----
    if probs.dim() == 1:
        # grab the one threshold (first entry)…
        if isinstance(thresholds, dict):
            thresh = next(iter(thresholds.values()))
        elif isinstance(thresholds, np.ndarray):
            thresh = thresholds.flat[0]
        else:
            raise ValueError(f"Unsupported thresholds type {type(thresholds)} for 1‑D probs")
        # …and return a 1‑D binary vector
        return (probs >= float(thresh)).float()

    # ---- 2) multi‑class case: probs dim == 2 ----
    if probs.dim() == 2:
        preds = torch.zeros_like(probs)
        if isinstance(thresholds, dict):
            for class_idx, (key, thresh) in enumerate(thresholds.items()):
                preds[:, class_idx] = (probs[:, class_idx] >= float(thresh)).float()
        elif isinstance(thresholds, np.ndarray):
            for class_idx, thresh in enumerate(thresholds):
                preds[:, class_idx] = (probs[:, class_idx] >= float(thresh)).float()
        else:
            raise ValueError(f"Unsupported thresholds type {type(thresholds)} for 2‑D probs")
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
    """
    # Convert to numpy arrays if necessary
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Ensure output directory exists
    save_dir = os.path.join(save_dir, 'roc_plots')
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


def plot_prediction_distributions(model_probs:list, 
                                  tasks:list, 
                                  class_idx:int=None, 
                                  sample_idx:int=None, 
                                  bins:int=20, 
                                  kde:bool=False, 
                                  model_names:list=None,
                                  save_dir:str='results/plots'):
    """
    Plot the distribution of model prediction values for a given class or sample.
    Args:
        model_probs (list of np.ndarray): Each element is an array of shape (N, C) 
            with probabilities or logits from one model for N samples and C classes.
        tasks (list of str): Names of the classes corresponding to columns in model_probs.
        class_idx (int, optional): Index of the class to analyze. If provided, will plot 
            distributions of that class's predictions across models.
        sample_idx (int, optional): Index of a specific sample to analyze. If provided, 
            will plot distribution of that sample's predictions across models.
        bins (int): Number of bins for histogram (if kde=False).
        kde (bool): If True, plot kernel density estimate curves instead of histogram.
        model_names (list of str, optional): Names of models for legend.
        save_dir (str): Path to directory where plots should be saved
    """
    # Prepare output directory
    if kde:
        out_dir = os.path.join(save_dir, 'prediction_distribution_kde')
        os.makedirs(out_dir, exist_ok=True)
    if not kde:
        out_dir = os.path.join(save_dir, 'prediction_distribution_hist')
        os.makedirs(out_dir, exist_ok=True)
    
    # First, convert everything to numpy arrays to avoid Tensor methods in plotting
    np_preds = []
    for p in model_probs:
        if torch.is_tensor(p):
            np_preds.append(p.detach().cpu().numpy())
        else:
            np_preds.append(np.array(p))

    # If neither index is given, iterate over all classes
    if class_idx is None and sample_idx is None:
        for idx, cls in enumerate(tasks):
            plot_prediction_distributions(
                model_probs=np_preds,
                tasks=tasks,
                class_idx=idx,
                sample_idx=None,
                bins=bins,
                kde=kde,
                model_names=model_names,
                save_dir=save_dir
            )
        return

    # --- CLASS‐WISE DISTRIBUTION ---
    if class_idx is not None:
        class_name = tasks[class_idx]
        plt.figure(figsize=(6,4))

        for i, preds in enumerate(np_preds):
            values = preds[:, class_idx]
            label = model_names[i] if model_names else f"Model_{i+1}"
            if kde:
                sns.kdeplot(values, label=label, shade=None)
            else:
                plt.hist(values, bins=bins, alpha=0.4, density=True, label=label)

        plt.title(f"Distribution of predictions for class: {class_name}")
        plt.xlabel("Predicted value" + (" (logit)" if (values.min()<0 or values.max()>1) else " (probability)"))
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        # save and close
        fname = f"distribution_{class_name.replace(' ','_')}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()
        return

    # --- SAMPLE‐WISE COMPARISON ---
    if sample_idx is not None:
        plt.figure(figsize=(6,4))
        for model_nr, preds in enumerate(np_preds):
            vals = preds[sample_idx]
            label = model_names[model_nr] if model_names else f"Model_{model_nr+1}"
            plt.plot(range(len(vals)), vals, marker='o', label=label)

        plt.title(f"Model predictions for sample {sample_idx}")
        plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
        plt.ylabel("Predicted probability")
        plt.legend()
        plt.tight_layout()

        # save and close
        fname = f"sample_{sample_idx}_preds.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()
        return

    # Fallback
    raise ValueError("Please specify either class_idx or sample_idx (or neither to plot all classes).")



def plot_model_correlation(model_probs: list, model_names: list = None, save_dir:str=None):
    """
    Plot a heatmap of Pearson correlation coefficients between each pair of models' predictions.
    Args:
        model_probs (list of np.ndarray): List of model prediction arrays of shape (N, C).
            All models should have predictions for the same N samples and C classes.
        model_names (list of str, optional): Names of the models for labeling axes.
        save_dir (str): Path to directory where plots should be saved.
    """
    out_dir = os.path.join(save_dir, 'model_correlation')
    os.makedirs(out_dir, exist_ok=True)

        # First, convert everything to numpy arrays to avoid Tensor methods in plotting
    np_preds = []
    for p in model_probs:
        if torch.is_tensor(p):
            np_preds.append(p.detach().cpu().numpy())
        else:
            np_preds.append(np.array(p))

    # Flatten each model's predictions into a 1D array (concatenate all class predictions for all samples)
    flat_preds = [pred.reshape(-1) for pred in np_preds]
    M = len(flat_preds)
    if M == 0:
        return
    
    # Compute correlation matrix (MxM)
    # np.corrcoef expects each row as a variable (model) and columns as observations
    corr_matrix = np.corrcoef(flat_preds)
    # Plot heatmap
    plt.figure(figsize=(5,4))
    im = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # Configure tick labels
    labels = model_names if model_names is not None else [f"M{i+1}" for i in range(M)]
    plt.xticks(range(M), labels, rotation=45, ha='right')
    plt.yticks(range(M), labels)
    plt.title("Correlation of Model Prediction Outputs")
    # Optionally annotate correlation values on the heatmap
    for i in range(M):
        for j in range(M):
            plt.text(j, i, f"{corr_matrix[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
    plt.tight_layout()
    
    fname = f"correlation_matrix.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()



def plot_umap_model_predictions(model_probs: list,
                                 model_names: list = None,
                                 n_neighbors: int = 15,
                                 min_dist: float = 0.1,
                                 metric: str = 'euclidean',
                                 n_components: int = 2,
                                 save_dir: str = 'results/plots/umap'):
    """
    Apply UMAP to the set of model prediction vectors and save a scatter plot
    in 2D or 3D, colored by model.

    Args:
        model_probs (list of np.ndarray or torch.Tensor): Each element is shape (N, C).
        model_names (list of str, optional): Names of the models, length == len(model_probs).
        n_neighbors (int): UMAP `n_neighbors`.
        min_dist (float): UMAP `min_dist`.
        metric (str): UMAP distance metric (e.g. 'euclidean', 'cosine').
        n_components (int): 2 or 3. Dimension of the embedding.
        save_dir (str): Directory under which to save the plot.
    """
    out_dir = os.path.join(save_dir, 'umap')
    os.makedirs(out_dir, exist_ok=True)

    # 1) Prepare data
    mats = []
    for p in model_probs:
        if hasattr(p, 'detach'):
            mats.append(p.detach().cpu().numpy())
        else:
            mats.append(np.array(p))
    if not mats:
        raise ValueError("No model predictions provided.")

    # stack into (M*N, C) and label each row with its model index
    n_models = len(mats)
    n_samples, n_dims = mats[0].shape
    X = np.vstack(mats)                           # shape = (M * N, C)
    labels = np.repeat(np.arange(n_models), n_samples)

    # 2) Run UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   metric=metric,
                   n_components=n_components)
    embedding = reducer.fit_transform(X)          # shape = (M*N, n_components)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(n_models):
            mask = labels == i
            name = model_names[i] if model_names else f"Model_{i+1}"
            ax.scatter(embedding[mask, 0],
                       embedding[mask, 1],
                       s=5, alpha=0.6,
                       label=name)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("UMAP of Model Prediction Vectors (2D)")
        legend = ax.legend(loc="best", markerscale=3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "umap_model_comparison_2d.png")
        fig.savefig(out_path, dpi=300)

    elif n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n_models):
            mask = labels == i
            name = model_names[i] if model_names else f"Model_{i+1}"
            ax.scatter(embedding[mask, 0],
                       embedding[mask, 1],
                       embedding[mask, 2],
                       s=5, alpha=0.6,
                       label=name)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_zlabel("UMAP3")
        ax.set_title("UMAP of Model Prediction Vectors (3D)")
        legend = ax.legend(loc="best", markerscale=3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "umap_3d.html")
        
        # interactive 3D
        fig = px.scatter_3d(
            x=embedding[:,0], y=embedding[:,1], z=embedding[:,2],
            color=labels, opacity=0.7,
            title="3D UMAP (Plotly)"
        )
        fig.write_html(out_path, include_plotlyjs='cdn')

    else:
        raise ValueError("n_components must be 2 (for 2D) or 3 for (3D -> interactive html).")

    print(f"Saved UMAP ({n_components}D) to {out_path}")


def compute_confusion_matrix_metrics(binary_preds: np.ndarray,
                                     targets: np.ndarray,
                                     tasks: list) -> pd.DataFrame:
    """
    Compute confusion matrix and derived metrics for each class in a multi-label setting.
    
    Parameters
    ----------
    binary_preds : (N, C) np.ndarray
        Final 0/1 predictions after thresholding.
    targets : (N, C) np.ndarray
        Ground-truth binary labels.
    tasks : list[str]
        All class names; defines column order in arrays.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rows for each task containing:
        - TP, TN, FP, FN (confusion matrix components)
        - Sensitivity (Recall/TPR)
        - Specificity (TNR)
        - Precision (PPV)
        - NPV (Negative Predictive Value)
        - F1-Score
        - Accuracy
        - Prevalence
    """
    metrics_list = []
    
    for task_idx, task in enumerate(tasks):
        y_true = targets[:, task_idx]
        y_pred = binary_preds[:, task_idx]
        
        # Compute confusion matrix components
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        
        # Compute derived metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall/TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # PPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0         # Negative Predictive Value
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        prevalence = (tp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        metrics_list.append({
            'Task': task,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Sensitivity (Recall)': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'Precision (PPV)': round(precision, 4),
            'NPV': round(npv, 4),
            'F1-Score': round(f1, 4),
            'Accuracy': round(accuracy, 4),
            'Prevalence': round(prevalence, 4)
        })
    
    df = pd.DataFrame(metrics_list)
    return df
