import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load configuration and prepare data
with open('config.json', 'r') as f:
    config = json.load(f)
ensemble_cfg = config['ensemble']
eval_cfg = config['evaluation']

# List of class names (tasks) and the subset of classes to evaluate (for mean F1)
# If not explicitly provided in config, we use the CheXpert 14-class order.
tasks = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
eval_tasks = eval_cfg.get('evaluation_sub_tasks', [])  # e.g. 5 target classes

# Load model predictions on the validation set.
# Assuming we have model objects or precomputed outputs for each model:
model_probs = []      # list of numpy arrays, each shape (N_val, num_classes)
ground_truth = None   # numpy array of shape (N_val, num_classes)
for model_cfg in config['models']:
    # Example: load or compute validation predictions for each model
    # Here we assume a function model_predict(model_cfg) that returns (probs, gt)
    probs, gt = model_predict(model_cfg, validation=True)
    model_probs.append(probs)           # probs as numpy array (N_val, C)
    if ground_truth is None:
        ground_truth = gt               # ground truth labels (N_val, C)
# Ensure all model_probs are numpy arrays for efficiency
model_probs = [np.asarray(p) for p in model_probs]
gt_labels = np.asarray(ground_truth)

# 2. Load distinctiveness values and build base weight matrix
distinct_files = ensemble_cfg.get('distinctiveness_files', [])
distinct_vals_list = []
for path in distinct_files:
    with open(path, 'r') as f:
        dist_dict = json.load(f)
        # Fix any known typos in class names (e.g., "Pleaural Effusion" -> "Pleural Effusion")
        if 'Pleaural Effusion' in dist_dict:
            dist_dict['Pleural Effusion'] = dist_dict.pop('Pleaural Effusion')
        distinct_vals_list.append(dist_dict)

num_models = len(distinct_vals_list)
num_classes = len(tasks)
# Initialize weight matrix (models x classes) and fill with distinctiveness values
weight_matrix = np.ones((num_models, num_classes), dtype=np.float32)
for i, dist_dict in enumerate(distinct_vals_list):
    for cls_name, dist_val in dist_dict.items():
        if cls_name in tasks:
            j = tasks.index(cls_name)
            weight_matrix[i, j] = dist_val
# Normalize weights so that for each class (column) the sum across models = 1
col_sums = weight_matrix.sum(axis=0, keepdims=True)
# Avoid division by zero in case of any zero-sum column
col_sums[col_sums == 0] = 1e-8
weight_matrix = weight_matrix / col_sums  # shape: (M, C)


# 3. Apply threshold tuning before ensemble if specified
tune_stage = ensemble_cfg.get('threshold_tuning', {}).get('stage', 'none')
tune_metric = ensemble_cfg.get('threshold_tuning', {}).get('metric', 'f1')
# We will perform threshold tuning on the validation set if required.
model_binary_preds = []  # will hold binary predictions per model (after thresholding if pre)
if tune_stage == 'pre':
    # Compute optimal threshold for each model (per class) on validation set - normal threshold tuning
    for i, probs in enumerate(model_probs):
        # If needed, aggregate probabilities per study (taking max over views)
        # For example, using a helper similar to get_max_prob_per_view:
        probs_val = probs  # shape (N_val, C), assuming one sample per patient or already aggregated
        gt_val = gt_labels
        # Find optimal thresholds for this model’s probabilities
        opt_thresholds = {}
        for c, cls in enumerate(tasks):
            # Search threshold from 0 to 1 by 0.01 for max F1
            best_t = 0.5
            best_f1 = -1.0
            y_true = gt_val[:, c]
            for t in np.arange(0.0, 1.001, 0.01):
                y_pred = (probs_val[:, c] >= t).astype(int)
                # Compute F1 for this class (avoid division by 0 issues)
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            opt_thresholds[cls] = best_t
        # Apply the thresholds to get binary predictions for this model
        binary = (probs_val >= np.array([opt_thresholds[cls] for cls in tasks])).astype(np.int8)
        model_binary_preds.append(binary)
    model_binary_preds = np.stack(model_binary_preds, axis=0)  # shape: (M, N_val, C)
else:
    # No pre-ensemble thresholding: just use raw probabilities (to be thresholded post-ensemble)
    model_binary_preds = np.stack(model_probs, axis=0)  # shape: (M, N_val, C)


#************************* Till here nothing new

# 4. Grid search over a and b
a_values = np.arange(0.01, 5.01, 0.01)
b_values = np.arange(0.01, 5.01, 0.01)
mean_f1_grid = np.zeros((len(a_values), len(b_values)), dtype=np.float32)

# Ground truth at evaluation level (if multiple images per patient, aggregate by patient)
# If needed, perform the same max-per-study aggregation on gt_labels (here assumed one-to-one for simplicity)
eval_gt = gt_labels  # shape (N_val, C) ground truth for evaluation

# Iterate over the grid of (a, b)
for ia, a in enumerate(a_values):
    # To improve efficiency, compute weight_matrix^b once per b in the inner loop
    for ib, b in enumerate(b_values):
        # Compute adjusted weights: a * (base_weight_matrix ** b)
        W_ab = a * (weight_matrix ** b)  # shape: (M, C)
        # Combine model predictions with these weights
        # If pre-thresholded, model_binary_preds contains 0/1; if post, contains probabilities
        # Multiply each model's predictions by its weights per class
        # Expand W_ab to (M, 1, C) to broadcast across N samples
        weighted_preds = W_ab[:, None, :] * model_binary_preds  # shape: (M, N_val, C)
        ensemble_scores = weighted_preds.sum(axis=0)            # shape: (N_val, C)
        # If post-thresholding is needed, find thresholds for ensemble (optional due to cost)
        if tune_stage == 'post':
            # (Optionally tune threshold for ensemble output here. To avoid heavy computation 
            # for each (a,b), we use a fixed 0.5 threshold for evaluation in this grid search.)
            ensemble_binary = (ensemble_scores >= 0.5).astype(int)
        else:
            # If already pre-thresholded (votes), treat >=0.5 of weighted vote as positive
            ensemble_binary = (ensemble_scores >= 0.5).astype(int)
        # Compute per-class F1 scores on validation
        f1_scores = []
        for cls in eval_tasks if eval_tasks else tasks:
            c_idx = tasks.index(cls)
            y_true = eval_gt[:, c_idx]
            y_pred = ensemble_binary[:, c_idx]
            # Compute F1 for class c_idx
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        mean_f1 = np.mean(f1_scores)
        mean_f1_grid[ia, ib] = mean_f1

# 5. Identify best (a, b) and visualize results
best_idx = np.unravel_index(np.argmax(mean_f1_grid), mean_f1_grid.shape)
best_a = a_values[best_idx[0]]
best_b = b_values[best_idx[1]]
best_score = mean_f1_grid[best_idx]
print(f"Best mean F1 = {best_score:.4f} at a = {best_a}, b = {best_b}")

# Plot heatmap of F1 scores over the grid
plt.figure(figsize=(8, 6))
sns.heatmap(mean_f1_grid, xticklabels=20, yticklabels=20, cmap="viridis")
plt.title("Mean F1 Score on Validation Set")
plt.xlabel("a value")
plt.ylabel("b value")
# (Optionally adjust tick labels to actual values if needed)
plt.show()
