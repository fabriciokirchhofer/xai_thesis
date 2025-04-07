from sklearn import metrics
from sklearn.metrics import f1_score, auc, roc_curve
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd



#******************** Utils ********************
#******************** preprocessing ********************
def remove_prefix(dict, prefix):
    """
    Function to remove additional prefix created while saving the model
    should be sth like: "model.features.norm0.weight" but is currently "model.model.features.norm0.weight"
    """
    new_state_dict = {}
    for key, value in dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


# def extract_study_id(mode):
#     """
#     Args:
#         mode (str): either 'val' or 'test'. 
#         It will then go either to the validation or test csv file
#         and extract from there the study id.

#     Return: DataFrame with the study id

#     """
#     if mode == 'val':
#         df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/valid.csv')
#         parts = df.split('/')
#         agg =  '/'.join(parts[:3])
#         df['study_id'] = df.iloc[:,0].apply(agg)
#         return df
    
#     elif mode == 'test':
#         df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/test.csv')
#         parts = df.split('/')
#         agg =  '/'.join(parts[:3])
#         df['study_id'] = df.iloc[:,0].apply(agg)
#         return df
#     else:
#         raise ValueError(f"Expected either 'val' or 'test' tasks, but got sth else.")



def extract_study_id(mode):
    """
    Args:
        mode (bool): either False goes to 'val' and True goes to 'test'. 
            It will then read the corresponding CSV file and extract the study ID
            from the image file paths (assuming the study ID is encoded in the first
            three parts of the file path, separated by '/').
    Returns:
        DataFrame: The CSV DataFrame with an added 'study_id' column.
    """
    if mode == False:
        df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/valid.csv')
        # Apply lambda to the first column to extract the study id.
        df['study_id'] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[:3]))
        return df

    elif mode:
        df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/test.csv')
        df['study_id'] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[:3]))
        return df

    else:
        raise ValueError("Expected either 'val' or 'test' mode, but got something else.")

#******************** Evaluation ********************


def compute_accuracy(predictions, labels):
    """
    Computes overall binary accuracy across all tasks.
    """
    correct = (predictions == labels).float().sum()
    total = labels.numel()
    accuracy = correct / total
    return accuracy.item()


def comput_youden_idx(ground_truth, preds, tasks):
    youden_idx = {}
    for i, task in enumerate(tasks):
        pred = preds[:, i]
        # Compute confusion matrix components
        tp = ((ground_truth[:, i] == 1) & (pred == 1)).sum()
        fn = ((ground_truth[:, i] == 1) & (pred == 0)).sum()
        tn = ((ground_truth[:, i] == 0) & (pred == 0)).sum()
        fp = ((ground_truth[:, i] == 0) & (pred == 1)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = sensitivity + specificity - 1
        youden_idx[task] = score
    return youden_idx


def compute_f1_score(ground_truth, preds, tasks):
    f1_scores = {}
    for i, task in enumerate(tasks):
        pred = preds[:, i]
        score = f1_score(ground_truth[:, i], pred, zero_division=0.0)
        f1_scores[task] = score
    return f1_scores


def auroc(predictions, ground_truth, tasks, n_classes=14):
    """
    Computes per-class AUROC using continuous probabilities.
    Args:
        predictions (array like): of shape (n_samples,) or (n_samples, n_classes)
        ground_truth (array like): of shape (n_samples,) or (n_samples, n_classes)
        tasks (list of str): Names for each class/task.
        n_classes (int): must match the number of classes. By default 14
    
    Returns:
        dict: Mapping from task name to its ROC AUC score.

    """
    # Check format of predictions
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    elif not isinstance(predictions, np.ndarray):       
        predictions = np.ndarray(predictions)
        predictions = np.array(predictions).reshape(-1,n_classes)

    # Check format of gt labels
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()
    elif not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)

    # Catch mismatch of classes and tasks (model dependent)
    if len(tasks) != n_classes:
        raise ValueError(f"Expected {n_classes} tasks, but got {len(tasks)}")
    
    auc_results = {}
    for i, task in enumerate(tasks):
        gt = ground_truth[:, i]
        pred = predictions[:, i]
        
        # Check if there are both positive and negative samples.
        # Here imbalances show up (e.g. fractures)
        if len(np.unique(gt)) < 2:
            print(f"Warning: Only one class present in ground truth for {task}. Skipping ROC AUC computation.")
            auc_results[task] = float('nan')
        else:
            auc_results[task] = metrics.roc_auc_score(y_true=gt, y_score=pred)

    return auc_results


def find_optimal_thresholds(probabilities, ground_truth, tasks, step=0.01, metric="f1"):
    """
    Finds optimal threshold for each class based on maximizing the F1 score.

    Args:
        probabilities (numpy array): Array of shape (n_samples, n_classes) containing probabilities.
        ground_truth (numpy array): Array of shape (n_samples, n_classes) with true binary labels.
        tasks (list of str): List of class names.
        step (float): Step size for threshold search.

    Returns:
        optimal_thresholds (dict): Mapping from task to optimal threshold.
        metric_score_dict (dict): Mapping from task to the achieved score ().
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


def threshold_based_predictions(probs, thresholds, tasks):
    """
    Applies task-specific thresholds to probabilities to generate binary predictions.
    
    Args:
        probs (Tensor): shape (N, C), predicted probabilities
        thresholds (float or dict): single threshold or per-task threshold dict
        tasks (List[str]): list of task names
    
    Returns:
        Tensor: binary predictions (0.0 or 1.0), shape (N, C)
    """
    if isinstance(thresholds, dict):
        predictions = torch.zeros_like(probs)
        for i, task in enumerate(tasks):
            predictions[:, i] = (probs[:, i] >= thresholds[task]).float()
    else:
        predictions = (probs >= thresholds).float()
    return predictions


# ********************************* PLOTS *********************************

def plot_roc(predictions, ground_truth, tasks, n_classes=14):
        

        """
        Computes per-class AUROC using continuous probabilities.
        Args:
            predictions (array like): of shape (n_samples,) or (n_samples, n_classes)
            ground_truth (array like): of shape (n_samples,) or (n_samples, n_classes)
            tasks (list of str): Names for each class/task.
            n_classes (int): must match the number of classes. By default 14

        """
        print("Entered function plot")


        # Convert to numpy arrays if necessary
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.detach().cpu().numpy()
            
        if len(tasks) != n_classes:
            raise ValueError(f"Expected {n_classes} tasks, but got {len(tasks)}")

        # Create a figure for the combined ROC curves
        fig_combined, ax_combined = plt.subplots(figsize=(8, 6))

        for i, task in enumerate(tasks):
            gt = ground_truth[:, i]
            pred = predictions[:, i]
            
            # Skip tasks where only one class is present in the ground truth
            if len(np.unique(gt)) < 2:
                print(f"Warning: Only one class present in ground truth for {task}. Skipping ROC curve plot for this task.")
                continue
            
            # calculate FP and TP rates
            fpr, tpr, _ = roc_curve(gt, pred)
            roc_auc = auc(fpr, tpr)
            
            print(f"looping over {task}")
            # Plot individual ROC curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve for {task}')
            ax.legend(loc="lower right")
            fig.savefig('results/plots/roc_' + str(task) + '.png')
            
            # Add to combined plot
            ax_combined.plot(fpr, tpr, lw=2, label=f'{task} (AUC = {roc_auc:.2f})', linewidth=0.8)

        ax_combined.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
        ax_combined.set_xlim([0, 1])
        ax_combined.set_ylim([0, 1.05])
        ax_combined.set_xlabel('False Positive Rate')
        ax_combined.set_ylabel('True Positive Rate (sensitivity)')
        ax_combined.set_title('Combined ROC Curves for All Tasks')
        ax_combined.legend(loc="lower right", fontsize='small')
        fig_combined.savefig('results/plots/combined_roc.png')
