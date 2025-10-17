#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare class-wise distinctiveness-based normalized weights against Optuna-optimized weights.

Usage:
  python weight_comparison_plot.py \
      --distinct-root "/path/to/distinctiveness_root_folder" \
      --optuna-json "/path/to/optimized_weights.json" \
      --outdir "/path/to/output_dir"

Notes:
- The script finds distinctiveness JSON files and orders them as ModelA..ModelE using a filename-based ranking:
    * Filenames containing "ignore_<N>" are ranked by N (to map to ModelA, ModelB, etc.).
    * Filenames containing "ep<N>" are ranked after "ignore_*" files, by N.
    * Adjust `rank_key()` if naming differs.
- Only the five final evaluation classes are considered: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
- Distinctiveness values for each class are normalized across models to sum to 1 (producing weights per model).
- The script generates grouped bar plots per class comparing:
    (a) Distinctiveness-based normalized weights
    (b) Optuna-optimized weights (from the provided JSON).
- Plot details:
    * X-axis: Models (ModelA to ModelE)
    * Y-axis: Normalized Weight
    * Two bars per model (distinctiveness vs optuna)
    * A horizontal dashed line at 0.2 indicating the baseline (equal weight if all 5 models were equal).
- Plots are saved as PNG files in an output subdirectory (e.g., "outdir/bar_plots/").
"""

import os
import re
import json
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

# ---------- Helpers ----------

MODEL_LABELS = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"]

CANON_MAP = {
    "pleural effusion": "Pleural Effusion",
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
}

def canon_class(name: str) -> str:
    """Canonicalize a class name (handles case and common synonyms)."""
    key = name.strip().lower()
    return CANON_MAP.get(key, name.strip())

_ignore_re = re.compile(r"ignore[_\-]?(\d+)", re.IGNORECASE)
_ep_re     = re.compile(r"ep[_\-]?(\d+)", re.IGNORECASE)

def rank_key(path: str) -> tuple:
    """
    Produce a sortable key from a filename to map files to ModelA..ModelE in a stable order.
    Priority:
      1) Filenames containing "ignore_<N>" -> (0, N)
      2) Filenames containing "ep<N>"      -> (1, N)
      3) Others (fallback)                 -> (9, filename alphabetically)
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
    # Fallback: rank by filename alphabetically
    return (9, fname.lower())

def find_distinctiveness_files(root: str):
    """Return a sorted list of file paths ending with 'class_wise_distinctiveness.json'."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("class_wise_distinctiveness.json"):
                hits.append(os.path.join(dirpath, fn))
    hits.sort(key=rank_key)
    return hits

def load_classwise_json(path: str) -> dict:
    """Load a JSON file mapping class->value, and canonicalize class names."""
    with open(path, "r") as f:
        data = json.load(f)
    fixed = {}
    for k, v in data.items():
        fixed[canon_class(k)] = float(v)
    return fixed

def align_classes(model_dicts: list):
    """
    Given a list of per-model distinctiveness dicts (each {class: value}),
    ensure they share the same class set. Missing entries are filled with 0.0.
    Returns: (aligned_list, all_classes_list)
    """
    all_classes = set()
    for d in model_dicts:
        all_classes |= set(d.keys())
    all_classes = sorted(all_classes)
    aligned = []
    for d in model_dicts:
        aligned.append({cls: float(d.get(cls, 0.0)) for cls in all_classes})
    return aligned, all_classes

def normalize_distinctiveness_to_weights(model_dicts: list, classes: list) -> dict:
    """
    Normalize distinctiveness scores to weights per class (so weights sum to 1 across models for each class).
    Returns a dict: {class_name: [w_ModelA, w_ModelB, ..., w_ModelE]}.
    """
    weights = {}
    for cls in classes:
        # Gather scores for this class from each model (0.0 if missing)
        scores = [model_dicts[i].get(cls, 0.0) for i in range(len(model_dicts))]
        total = float(sum(scores))
        if total <= 0:
            weights[cls] = [0.0 for _ in scores]
        else:
            weights[cls] = [s / total for s in scores]
    return weights

def load_optuna_weights(optuna_json_path: str, classes: list, model_labels=None) -> dict:
    """
    Load an Optuna weights JSON and return {class: [w_ModelA, ..., w_ModelE]} for the specified classes.
    If model_labels (list of model names) is provided, use that ordering; otherwise, use ModelA..ModelE.
    """
    with open(optuna_json_path, "r") as f:
        optuna_data = json.load(f)
    wraw = optuna_data.get("Weights", {})
    # Canonicalize class names in the weights structure
    for mdl, cls_dict in list(wraw.items()):
        fixed = {}
        for k, v in cls_dict.items():
            fixed[canon_class(k)] = float(v)
        wraw[mdl] = fixed
    order = model_labels if model_labels is not None else MODEL_LABELS
    out = {}
    for cls in classes:
        out[cls] = [float(wraw.get(m, {}).get(cls, 0.0)) for m in order]
    return out

def ensure_outdir(path: str):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

# ---------- Plotting ----------

def plot_grouped_bars(distinct_w: dict, optuna_w: dict, model_labels: list, outdir: str, baseline: float = 0.2):
    """
    Generate a grouped bar plot for each class comparing distinctiveness vs optuna weights.
    Saves each plot as a PNG file in the given output directory.
    """
    ensure_outdir(outdir)
    x = np.arange(len(model_labels))
    bar_width = 0.4
    for cls in distinct_w:
        dw = distinct_w[cls]
        ow = optuna_w.get(cls, [0.0] * len(model_labels))
        plt.figure(figsize=(6, 4))
        # Plot bars for distinctiveness and optuna weights
        plt.bar(x - bar_width/2, dw, bar_width, label="Distinctiveness-based")
        plt.bar(x + bar_width/2, ow, bar_width, label="Optuna-optimized")
        # Baseline line at 0.2 (dashed)
        plt.axhline(baseline, color="gray", linestyle="--", linewidth=1, zorder=3)
        # Labels and title
        plt.xticks(x, model_labels)
        plt.xlabel("Model")
        plt.ylabel("Normalized Weight")
        plt.ylim(0, 1.0)
        plt.title(f"Optuna vs Distinctiveness Weights — {cls}")
        plt.legend(frameon=False)
        plt.tight_layout()
        # Save plot
        filename = f"{cls.replace(' ', '_')}_weights_comparison.png"
        plt.savefig(os.path.join(outdir, filename), dpi=300)
        plt.close()

# ---------- Main ----------

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
    
    parser.add_argument("--baseline", type=float, default=0.2,
                        help="Baseline weight for reference line (default=0.2)")
    args = parser.parse_args()
    
    ensure_outdir(args.outdir)

    ensure_outdir(args.outdir)
    bar_plot_dir = os.path.join(args.outdir, "bar_plots")
    ensure_outdir(bar_plot_dir)

    # 1. Collect distinctiveness files and map to ModelA..ModelE
    files = find_distinctiveness_files(args.distinct_root)
    if len(files) < 5:
        print(f"[WARN] Found {len(files)} distinctiveness files (expected >= 5). Proceeding with available files.")
    selected_files = files[:len(MODEL_LABELS)]
    model_to_file = OrderedDict()
    for label, fpath in zip(MODEL_LABELS, selected_files):
        model_to_file[label] = fpath
    print("Model file mapping:")
    for label, fpath in model_to_file.items():
        print(f"  {label} -> {fpath}")

    # 2. Load distinctiveness data for each model and align classes
    per_model_data = [load_classwise_json(fpath) for fpath in model_to_file.values()]
    aligned_data, all_classes = align_classes(per_model_data)
    # Only keep the five final evaluation classes
    final_classes = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    classes_to_use = [c for c in all_classes if c in final_classes]
    missing_classes = set(final_classes) - set(classes_to_use)
    if missing_classes:
        print(f"[WARN] The following final classes were not found in distinctiveness data: {sorted(missing_classes)}")
    # 3. Normalize distinctiveness values to weights per class
    distinct_w = normalize_distinctiveness_to_weights(aligned_data, classes_to_use)
    # 4. Load Optuna weights for these classes (aligned to models we have)
    model_list = list(model_to_file.keys())  # e.g., ModelA..ModelE (or fewer if limited files)
    optuna_w = load_optuna_weights(args.optuna_json, classes_to_use, model_labels=model_list)
    # 5. Plot grouped bar charts for each class
    plot_grouped_bars(distinct_w, optuna_w, model_list, bar_plot_dir, baseline=args.baseline)
    print(f"Bar plots saved to {bar_plot_dir}")

if __name__ == "__main__":
    main()












# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Analyze the transformation from class-wise distinctiveness values to final ensemble weights.

# Usage:
#   python distinct_to_ensemble_analysis.py \
#       --distinct-root "/path/to/distinctiveness_root_folder" \
#       --optuna-json   "/path/to/optimized_weights.json" \
#       --ensemble-json "/path/to/ensemble_matrix.json" \
#       --outdir        "/path/to/output_dir"

# Notes:
# - Recursively finds all "*class_wise_distinctiveness.json" files under the distinct-root and maps them to ModelA..ModelE using filename-based ranking:contentReference[oaicite:8]{index=8}.
# - Only the five evaluation classes are analyzed: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
# - Distinctiveness values are normalized into weights per class (summing to 1 across models):contentReference[oaicite:9]{index=9}.
# - Ensemble weights are read from ensemble_matrix.json (list of lists for each model) in the known class order and extracted for the five classes.
# - Produces bar plots comparing distinctiveness vs ensemble weights for each class, saved to the output directory.
# - (The optuna-json argument is accepted for completeness but not used in the analysis.)
# """
# import os
# import re
# import json
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import OrderedDict

# # Model labels assumed for up to 5 models
# MODEL_LABELS = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"]

# # Mapping to canonical class names for consistency (focus on five classes of interest)
# CANON_MAP = {
#     "pleural effusion": "Pleural Effusion",
#     "atelectasis": "Atelectasis",
#     "cardiomegaly": "Cardiomegaly",
#     "consolidation": "Consolidation",
#     "edema": "Edema"}

# def canon_class(name: str) -> str:
#     """Standardize class name casing/spelling to match expected names."""
#     key = name.strip().lower()
#     return CANON_MAP.get(key, name.strip())

# # Regex for ranking patterns
# _ignore_re = re.compile(r"ignore[_\-]?(\d+)", re.IGNORECASE)
# _ep_re     = re.compile(r"ep[_\-]?(\d+)", re.IGNORECASE)
# def rank_key(path: str) -> tuple:
#     """
#     Sort key for filenames to consistently map to ModelA..E:contentReference[oaicite:10]{index=10}.
#     Priority:
#       1) "...ignore_<N>..." -> (0, N)
#       2) "...ep<N>..."      -> (1, N)
#       3) others             -> (9, filename alphabetically)
#     """
#     fname = os.path.basename(path)
#     m_ign = _ignore_re.search(fname)
#     if m_ign:
#         try:
#             return (0, int(m_ign.group(1)))
#         except ValueError:
#             pass
#     m_ep = _ep_re.search(fname)
#     if m_ep:
#         try:
#             return (1, int(m_ep.group(1)))
#         except ValueError:
#             pass
#     return (9, fname.lower())

# def find_distinctiveness_files(root: str):
#     """Recursively find all 'class_wise_distinctiveness.json' files under root:contentReference[oaicite:11]{index=11}."""
#     hits = []
#     for dirpath, _, filenames in os.walk(root):
#         for fn in filenames:
#             if fn.endswith("class_wise_distinctiveness.json"):
#                 hits.append(os.path.join(dirpath, fn))
#     hits.sort()
#     return hits

# def load_classwise_json(path: str) -> dict:
#     """Load a class->value mapping from JSON and canonicalize class names:contentReference[oaicite:12]{index=12}."""
#     with open(path, "r") as f:
#         data = json.load(f)
#     fixed = {}
#     for k, v in data.items():
#         fixed[canon_class(k)] = float(v)
#     return fixed

# def align_classes(model_dicts: list):
#     """
#     Ensure all model dicts share the same class keys, filling missing entries with 0:contentReference[oaicite:13]{index=13}.
#     Returns a tuple: (aligned_dicts_list, class_list).
#     """
#     all_classes = set()
#     for d in model_dicts:
#         all_classes |= set(d.keys())
#     all_classes = sorted(all_classes)  # sort class names alphabetically
#     aligned = []
#     for d in model_dicts:
#         aligned.append({cls: float(d.get(cls, 0.0)) for cls in all_classes})
#     return aligned, all_classes

# def normalize_distinctiveness_to_weights(model_dicts: list, classes: list) -> dict:
#     """
#     Convert raw distinctiveness values into normalized weights per class (summing to 1):contentReference[oaicite:14]{index=14}.
#     Returns: {class_name: [w_ModelA, w_ModelB, ..., w_ModelE]}.
#     """
#     weights = {}
#     for cls in classes:
#         scores = [model_dicts[i].get(cls, 0.0) for i in range(len(model_dicts))]
#         total = float(sum(scores))
#         if total <= 0:
#             weights[cls] = [0.0] * len(scores)
#         else:
#             weights[cls] = [s / total for s in scores]
#     return weights

# def load_ensemble_weights(ensemble_json_path: str, class_names: list) -> dict:
#     """
#     Load the ensemble weight matrix and extract weights for the specified classes.
#     The ensemble JSON is expected to be a list of lists, one per model (ModelA..E),
#     with class weights in the predefined order of classes.
#     """
#     with open(ensemble_json_path, "r") as f:
#         matrix = json.load(f)
#     # Known class order in ensemble matrix (index mapping for classes)
#     class_order = [
#         "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
#         "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
#         "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
#         "Fracture", "Support Devices"
#     ]
#     # Build mapping from class name to index in ensemble matrix lists
#     class_to_index = {cls: idx for idx, cls in enumerate(class_order)}
#     # Extract weights for each class of interest across all models
#     ensemble_w = {}
#     num_models = len(matrix)
#     for cls in class_names:
#         if cls not in class_to_index:
#             # If class not found in known order (should not happen for our five classes)
#             continue
#         idx = class_to_index[cls]
#         # Collect the weight from each model's list at that index
#         values = []
#         for m in range(num_models):
#             # If a model list is shorter than expected index, skip or use 0.0
#             if idx < len(matrix[m]):
#                 values.append(float(matrix[m][idx]))
#             else:
#                 values.append(0.0)
#         ensemble_w[cls] = values
#     return ensemble_w

# def ensure_outdir(path: str):
#     """Create the directory if it doesn't exist."""
#     os.makedirs(path, exist_ok=True)

# def plot_bar_comparison(class_name: str, distinct_vals: list, ensemble_vals: list, outdir: str):
#     """
#     Create a side-by-side bar chart for one class, comparing distinctiveness vs ensemble weights.
#     Saves the plot as a PNG file in outdir.
#     """
#     # X positions for models
#     models = range(len(distinct_vals))
#     x = np.arange(len(distinct_vals))
#     width = 0.35  # width of each bar in a group
    
#     # Start a new figure for this class
#     fig, ax = plt.subplots(figsize=(6, 4))
#     rects1 = ax.bar(x - width/2, distinct_vals, width, label='Distinctiveness')
#     rects2 = ax.bar(x + width/2, ensemble_vals, width, label='Ensemble')
    
#     # Add labels and title
#     ax.set_ylabel("Normalized Weight")
#     ax.set_title(f"{class_name} – Distinctiveness vs Ensemble Weights")
#     ax.set_xticks(x)
#     # Use only as many MODEL_LABELS as we have models (in case <5 models found)
#     model_labels = MODEL_LABELS[:len(distinct_vals)]
#     ax.set_xticklabels(model_labels)
#     ax.legend()
    
#     # Annotate bar heights for clarity
#     for rect in rects1 + rects2:
#         height = rect.get_height()
#         ax.annotate(f"{height:.3f}",
#                     xy=(rect.get_x() + rect.get_width()/2, height),
#                     xytext=(0, 3), textcoords="offset points",
#                     ha='center', va='bottom')
    
#     fig.tight_layout()
#     # Save figure
#     ensure_outdir(outdir)
#     plot_path = os.path.join(outdir, f"bar_{class_name.replace(' ', '_')}.png")
#     plt.savefig(plot_path, dpi=300)
#     plt.close(fig)

# def main():
#     parser = argparse.ArgumentParser(description="Analyze distinctiveness vs ensemble weights")

#     parser.add_argument("--distinct-root", default="/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/distinctiveness_values/distinctiveness_cos_similarity_DeepLift_val_original_no_baseline_29.09.2025",
#                         help="Root folder to search for *class_wise_distinctiveness.json files")

#     parser.add_argument("--optuna-json", default="/home/fkirchhofer/repo/xai_thesis/optimized_weights_with_multivariate_sampler_dist_weighted_300.json",
#                         help="Path to optimized_weights.json (not used for computation)")

#     parser.add_argument("--ensemble-json", default="/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results/ensemble_DeepLift_original_no_baseline_29.09.2025/001_distinctiveness_weighted_val_5m_20250929_114345/distinctiveness_weight_matrix.json",
#                         help="Path to ensemble_matrix.json containing final ensemble weights")

#     parser.add_argument("--outdir", default="/home/fkirchhofer/repo/xai_thesis/AAA_evaluation_scripts/mapping_DeepLift_val_original_no_baseline",
#                         help="Output directory for visualizations and results")
#     args = parser.parse_args()
    
#     ensure_outdir(args.outdir)
    
#     # 1) Find and rank distinctiveness files
#     files = find_distinctiveness_files(args.distinct_root)
#     if len(files) < len(MODEL_LABELS):
#         print(f"[Warning] Found {len(files)} distinctiveness files (expected 5). Proceeding with available files.")
#     selected_files = files[:len(MODEL_LABELS)]  # take at most 5 files
#     model_to_file = OrderedDict()
#     for label, fpath in zip(MODEL_LABELS, selected_files):
#         model_to_file[label] = fpath
#     print("Model file mapping:")
#     for m, f in model_to_file.items():
#         print(f"  {m} -> {f}")
    
#     # 2) Load distinctiveness data for each model and align classes
#     per_model_data = []
#     for mdl in MODEL_LABELS[:len(selected_files)]:
#         per_model_data.append(load_classwise_json(model_to_file[mdl]))
#     aligned_dicts, all_classes = align_classes(per_model_data)
    
#     # 3) Filter to five evaluation classes of interest
#     eval_classes = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
#     classes_to_process = [c for c in all_classes if c in eval_classes]
#     classes_to_process.sort(key=lambda x: eval_classes.index(x))  # preserve the specified order
#     # (Alternatively, eval_classes is already alphabetical and can be used directly if all present)
    
#     # 4) Normalize distinctiveness values to get distinctiveness weights per class
#     distinct_w = normalize_distinctiveness_to_weights(aligned_dicts, classes_to_process)
    
#     # 5) Load ensemble weights and align to the same classes
#     ensemble_w = load_ensemble_weights(args.ensemble_json, classes_to_process)
    
#     # 6) Produce bar plots for each class
#     bar_dir = os.path.join(args.outdir, "bar_plots")
#     for cls in classes_to_process:
#         dw = distinct_w.get(cls, [])
#         ew = ensemble_w.get(cls, [])
#         if not dw or not ew:
#             continue  # skip if no data for class (should not happen for these five)
#         plot_bar_comparison(cls, dw, ew, bar_dir)
#     print(f"Saved bar plots for classes: {', '.join(classes_to_process)} -> {bar_dir}/")
    
#     # (Optional) You could add more analysis here, like scatter plots or difference plots as discussed.
    
# if __name__ == "__main__":
#     main()
