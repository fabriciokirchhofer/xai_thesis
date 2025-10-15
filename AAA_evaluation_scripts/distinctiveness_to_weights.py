#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the transformation from class-wise distinctiveness values to final ensemble weights.

Usage:
  python distinct_to_ensemble_analysis.py \
      --distinct-root "/path/to/distinctiveness_root_folder" \
      --optuna-json   "/path/to/optimized_weights.json" \
      --ensemble-json "/path/to/ensemble_matrix.json" \
      --outdir        "/path/to/output_dir"

Notes:
- Recursively finds all "*class_wise_distinctiveness.json" files under the distinct-root and maps them to ModelA..ModelE using filename-based ranking:contentReference[oaicite:8]{index=8}.
- Only the five evaluation classes are analyzed: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
- Distinctiveness values are normalized into weights per class (summing to 1 across models):contentReference[oaicite:9]{index=9}.
- Ensemble weights are read from ensemble_matrix.json (list of lists for each model) in the known class order and extracted for the five classes.
- Produces bar plots comparing distinctiveness vs ensemble weights for each class, saved to the output directory.
- (The optuna-json argument is accepted for completeness but not used in the analysis.)
"""
import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Model labels assumed for up to 5 models
MODEL_LABELS = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"]

# Mapping to canonical class names for consistency (focus on five classes of interest)
CANON_MAP = {
    "pleural effusion": "Pleural Effusion",
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
}
def canon_class(name: str) -> str:
    """Standardize class name casing/spelling to match expected names."""
    key = name.strip().lower()
    return CANON_MAP.get(key, name.strip())

# Regex for ranking patterns (as in correlation_calculator)
_ignore_re = re.compile(r"ignore[_\-]?(\d+)", re.IGNORECASE)
_ep_re     = re.compile(r"ep[_\-]?(\d+)", re.IGNORECASE)
def rank_key(path: str) -> tuple:
    """
    Sort key for filenames to consistently map to ModelA..E:contentReference[oaicite:10]{index=10}.
    Priority:
      1) "...ignore_<N>..." -> (0, N)
      2) "...ep<N>..."      -> (1, N)
      3) others             -> (9, filename alphabetically)
    """
    fname = os.path.basename(path)
    m_ign = _ignore_re.search(fname)
    if m_ign:
        try:
            return (0, int(m_ign.group(1)))
        except ValueError:
            pass
    m_ep = _ep_re.search(fname)
    if m_ep:
        try:
            return (1, int(m_ep.group(1)))
        except ValueError:
            pass
    return (9, fname.lower())

def find_distinctiveness_files(root: str):
    """Recursively find all 'class_wise_distinctiveness.json' files under root:contentReference[oaicite:11]{index=11}."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("class_wise_distinctiveness.json"):
                hits.append(os.path.join(dirpath, fn))
    hits.sort(key=rank_key)
    return hits

def load_classwise_json(path: str) -> dict:
    """Load a class->value mapping from JSON and canonicalize class names:contentReference[oaicite:12]{index=12}."""
    with open(path, "r") as f:
        data = json.load(f)
    fixed = {}
    for k, v in data.items():
        fixed[canon_class(k)] = float(v)
    return fixed

def align_classes(model_dicts: list):
    """
    Ensure all model dicts share the same class keys, filling missing entries with 0:contentReference[oaicite:13]{index=13}.
    Returns a tuple: (aligned_dicts_list, class_list).
    """
    all_classes = set()
    for d in model_dicts:
        all_classes |= set(d.keys())
    all_classes = sorted(all_classes)  # sort class names alphabetically
    aligned = []
    for d in model_dicts:
        aligned.append({cls: float(d.get(cls, 0.0)) for cls in all_classes})
    return aligned, all_classes

def normalize_distinctiveness_to_weights(model_dicts: list, classes: list) -> dict:
    """
    Convert raw distinctiveness values into normalized weights per class (summing to 1):contentReference[oaicite:14]{index=14}.
    Returns: {class_name: [w_ModelA, w_ModelB, ..., w_ModelE]}.
    """
    weights = {}
    for cls in classes:
        scores = [model_dicts[i].get(cls, 0.0) for i in range(len(model_dicts))]
        total = float(sum(scores))
        if total <= 0:
            weights[cls] = [0.0] * len(scores)
        else:
            weights[cls] = [s / total for s in scores]
    return weights

def load_ensemble_weights(ensemble_json_path: str, class_names: list) -> dict:
    """
    Load the ensemble weight matrix and extract weights for the specified classes.
    The ensemble JSON is expected to be a list of lists, one per model (ModelA..E),
    with class weights in the predefined order of classes.
    """
    with open(ensemble_json_path, "r") as f:
        matrix = json.load(f)
    # Known class order in ensemble matrix (index mapping for classes)
    class_order = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices"
    ]
    # Build mapping from class name to index in ensemble matrix lists
    class_to_index = {cls: idx for idx, cls in enumerate(class_order)}
    # Extract weights for each class of interest across all models
    ensemble_w = {}
    num_models = len(matrix)
    for cls in class_names:
        if cls not in class_to_index:
            # If class not found in known order (should not happen for our five classes)
            continue
        idx = class_to_index[cls]
        # Collect the weight from each model's list at that index
        values = []
        for m in range(num_models):
            # If a model list is shorter than expected index, skip or use 0.0
            if idx < len(matrix[m]):
                values.append(float(matrix[m][idx]))
            else:
                values.append(0.0)
        ensemble_w[cls] = values
    return ensemble_w

def ensure_outdir(path: str):
    """Create the directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def plot_bar_comparison(class_name: str, distinct_vals: list, ensemble_vals: list, outdir: str):
    """
    Create a side-by-side bar chart for one class, comparing distinctiveness vs ensemble weights.
    Saves the plot as a PNG file in outdir.
    """
    # X positions for models
    models = range(len(distinct_vals))
    x = np.arange(len(distinct_vals))
    width = 0.35  # width of each bar in a group
    
    # Start a new figure for this class
    fig, ax = plt.subplots(figsize=(6, 4))
    rects1 = ax.bar(x - width/2, distinct_vals, width, label='Distinctiveness')
    rects2 = ax.bar(x + width/2, ensemble_vals, width, label='Ensemble')
    
    # Add labels and title
    ax.set_ylabel("Normalized Weight")
    ax.set_title(f"{class_name} – Distinctiveness vs Ensemble Weights")
    ax.set_xticks(x)
    # Use only as many MODEL_LABELS as we have models (in case <5 models found)
    model_labels = MODEL_LABELS[:len(distinct_vals)]
    ax.set_xticklabels(model_labels)
    ax.legend()
    
    # Annotate bar heights for clarity
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    
    fig.tight_layout()
    # Save figure
    ensure_outdir(outdir)
    plot_path = os.path.join(outdir, f"bar_{class_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyze distinctiveness vs ensemble weights")

    parser.add_argument("--distinct-root", default="/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/distinctiveness_values/distinctiveness_cos_similarity_DeepLift_val_original_no_baseline_29.09.2025",
                        help="Root folder to search for *class_wise_distinctiveness.json files")

    parser.add_argument("--optuna-json", default="/home/fkirchhofer/repo/xai_thesis/optimized_weights_with_multivariate_sampler_dist_weighted_300.json",
                        help="Path to optimized_weights.json (not used for computation)")

    parser.add_argument("--ensemble-json", default="/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results/ensemble_DeepLift_original_no_baseline_29.09.2025/001_distinctiveness_weighted_val_5m_20250929_114345/distinctiveness_weight_matrix.json",
                        help="Path to ensemble_matrix.json containing final ensemble weights")

    parser.add_argument("--outdir", default="/home/fkirchhofer/repo/xai_thesis/AAA_evaluation_scripts/mapping_DeepLift_val_original_no_baseline",
                        help="Output directory for visualizations and results")
    args = parser.parse_args()
    
    ensure_outdir(args.outdir)
    
    # 1) Find and rank distinctiveness files
    files = find_distinctiveness_files(args.distinct_root)
    if len(files) < len(MODEL_LABELS):
        print(f"[Warning] Found {len(files)} distinctiveness files (expected 5). Proceeding with available files.")
    selected_files = files[:len(MODEL_LABELS)]  # take at most 5 files
    model_to_file = OrderedDict()
    for label, fpath in zip(MODEL_LABELS, selected_files):
        model_to_file[label] = fpath
    print("Model file mapping:")
    for m, f in model_to_file.items():
        print(f"  {m} -> {f}")
    
    # 2) Load distinctiveness data for each model and align classes
    per_model_data = []
    for mdl in MODEL_LABELS[:len(selected_files)]:
        per_model_data.append(load_classwise_json(model_to_file[mdl]))
    aligned_dicts, all_classes = align_classes(per_model_data)
    
    # 3) Filter to five evaluation classes of interest
    eval_classes = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    classes_to_process = [c for c in all_classes if c in eval_classes]
    classes_to_process.sort(key=lambda x: eval_classes.index(x))  # preserve the specified order
    # (Alternatively, eval_classes is already alphabetical and can be used directly if all present)
    
    # 4) Normalize distinctiveness values to get distinctiveness weights per class
    distinct_w = normalize_distinctiveness_to_weights(aligned_dicts, classes_to_process)
    
    # 5) Load ensemble weights and align to the same classes
    ensemble_w = load_ensemble_weights(args.ensemble_json, classes_to_process)
    
    # 6) Produce bar plots for each class
    bar_dir = os.path.join(args.outdir, "bar_plots")
    for cls in classes_to_process:
        dw = distinct_w.get(cls, [])
        ew = ensemble_w.get(cls, [])
        if not dw or not ew:
            continue  # skip if no data for class (should not happen for these five)
        plot_bar_comparison(cls, dw, ew, bar_dir)
    print(f"Saved bar plots for classes: {', '.join(classes_to_process)} -> {bar_dir}/")
    
    # (Optional) You could add more analysis here, like scatter plots or difference plots as discussed.
    
if __name__ == "__main__":
    main()
