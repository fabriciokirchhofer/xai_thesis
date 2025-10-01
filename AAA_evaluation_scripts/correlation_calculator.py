
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze correlation between distinctiveness-based weights and Optuna-optimized weights.

Usage:
  python weight_correlation_analysis.py \
      --distinct-root "/path/to/distinctiveness_root_folder" \
      --optuna-json "/path/to/optimized_weights.json" \
      --outdir "/path/to/output_dir"

Notes:
- The script walks the distinctiveness folder recursively and picks files named
  "*class_wise_distinctiveness.json".
- It orders them to ModelA..ModelE using a filename-based rank:
    * "...ignore_<N>..." => rank N
    * "...ep<N>..."      => rank 100 + N  (so ep1, ep2 follow after ignore_1..3)
  Adjust `rank_key()` if your naming changes.
- Distinctiveness normalization: per-class weights sum to 1 across models.
- Correlations: Pearson & Spearman per class.
- Plots: scatter per class + correlation heatmap.
"""

import os
import re
import json
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
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
    key = name.strip().lower()
    return CANON_MAP.get(key, name.strip())

_ignore_re = re.compile(r"ignore[_\-]?(\d+)", re.IGNORECASE)
_ep_re     = re.compile(r"ep[_\-]?(\d+)", re.IGNORECASE)

def rank_key(path: str) -> tuple:
    """
    Produce a sortable key from filename to map files to ModelA..E in a stable, intended order.

    Priority:
      1) "...ignore_<N>..."  -> (0, N)
      2) "...ep<N>..."       -> (1, N)
      3) fallback            -> (9, alphabetical)
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
    """Return a sorted list of class_wise_distinctiveness.json file paths."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("class_wise_distinctiveness.json"):
                hits.append(os.path.join(dirpath, fn))
    hits.sort(key=rank_key)
    return hits

def load_classwise_json(path: str) -> dict:
    """Load a class->value mapping and canonicalize class names."""
    with open(path, "r") as f:
        data = json.load(f)
    fixed = {}
    for k, v in data.items():
        fixed[canon_class(k)] = float(v)
    return fixed

def align_classes(model_dicts: list) -> list:
    """
    Given a list of per-model dicts {class: value}, ensure they share the same class set.
    Missing entries filled with 0.0.
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
    Per class, convert raw distinctiveness across models into weights summing to 1.
    Returns: {class: [wA, wB, wC, wD, wE]}
    """
    weights = {}
    for cls in classes:
        scores = [model_dicts[i][cls] for i in range(len(model_dicts))]
        total = float(sum(scores))
        if total <= 0:
            weights[cls] = [0.0 for _ in scores]
        else:
            weights[cls] = [s / total for s in scores]
    return weights

def load_optuna_weights(optuna_json_path: str, classes: list) -> dict:
    """
    Read optuna json and produce {class: [wA..wE]} aligned to MODEL_LABELS.
    """
    with open(optuna_json_path, "r") as f:
        optuna_data = json.load(f)

    # Expect top-level "Weights" with {ModelA: {class: w, ...}, ...}
    wraw = optuna_data.get("Weights", {})
    # Canonicalize class names inside
    for mdl in list(wraw.keys()):
        fixed = {}
        for k, v in wraw[mdl].items():
            fixed[canon_class(k)] = float(v)
        wraw[mdl] = fixed

    out = {}
    for cls in classes:
        out[cls] = [float(wraw[m].get(cls, 0.0)) for m in MODEL_LABELS]
    return out

def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)

# ---------- Plotting ----------

def plot_scatter_per_class(distinct_w: dict, optuna_w: dict, outdir: str, baseline: float = 0.2):
    ensure_outdir(outdir)
    for cls, dw in distinct_w.items():
        ow = optuna_w[cls]
        x = np.array(dw, dtype=float)
        y = np.array(ow, dtype=float)

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, marker="x", s=100)
        # annotate points with model labels
        for i, label in enumerate(MODEL_LABELS):
            plt.text(x[i] + 0.01, y[i] + 0.01, label, fontsize=9)

        # baseline lines for equal weight
        plt.axhline(baseline, color="gray", linestyle="--", linewidth=1)
        plt.axvline(baseline, color="gray", linestyle="--", linewidth=1)

        # best fit line (if variance exists)
        if np.std(x) > 0 and np.std(y) > 0:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(min(x.min(), 0), max(x.max(), 1), 100)
            plt.plot(xx, m * xx + b, color="red", linewidth=1)

        plt.xlabel("Distinctiveness-based weight")
        plt.ylabel("Optuna-optimized weight")
        plt.title(f"Optuna vs Distinctiveness — {cls}")
        plt.tight_layout()
        fn = os.path.join(outdir, f"scatter_{cls.replace(' ', '_')}.png")
        plt.savefig(fn, dpi=300)
        plt.close()

def plot_correlation_heatmap(distinct_w: dict, optuna_w: dict, outpath: str, corr_type: str = "pearson"):
    """
    Heatmap rows: distinctiveness per class; columns: optuna per class.
    corr_type: 'pearson' or 'spearman'
    """
    classes = list(distinct_w.keys())
    K = len(classes)
    mat = np.zeros((K, K), dtype=float)
    for i, ci in enumerate(classes):
        x = np.array(distinct_w[ci], dtype=float)
        for j, cj in enumerate(classes):
            y = np.array(optuna_w[cj], dtype=float)
            if corr_type == "spearman":
                c, _ = spearmanr(x, y)
            else:
                c, _ = pearsonr(x, y)
            mat[i, j] = c

    plt.figure(figsize=(1.2 * K, 1.0 * K))
    im = plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(K), classes, rotation=45, ha="right")
    plt.yticks(range(K), classes)

    # annotate cells
    for i in range(K):
        for j in range(K):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    plt.title(f"Correlation Heatmap ({corr_type.capitalize()})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distinct-root", default="/home/fkirchhofer/repo/xai_thesis/distinctiveness_cos_similarity_LRP_val_1280_24.09.2025",
                     help="Root folder to walk for *class_wise_distinctiveness.json files")
    
    ap.add_argument("--optuna-json", default="/home/fkirchhofer/repo/xai_thesis/optimized_weights_with_multivariate_sampler_dist_weighted_300.json",
                    help="Path to Optuna weights JSON")
    
    ap.add_argument("--outdir", default="weight_correlation_outputs", 
                    help="Output directory")
    
    ap.add_argument("--baseline", type=float, default=0.2, 
                    help="Baseline equal weight to show in scatter plots")
    
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    scatter_dir = os.path.join(args.outdir, "scatter_plots")
    ensure_outdir(scatter_dir)

    # 1) Find & order model distinctiveness files -> map to ModelA..E
    files = find_distinctiveness_files(args.distinct_root)
    if len(files) < 5:
        print(f"[WARN] Found {len(files)} distinctiveness files; expected >=5. Proceeding with what was found.")
    # Take first five by rank for ModelA..E (or adjust if you want all)
    selected = files[:len(MODEL_LABELS)]
    model_to_file = OrderedDict()
    for label, fpath in zip(MODEL_LABELS, selected):
        model_to_file[label] = fpath
    print("Model file mapping (auto-detected):")
    for k, v in model_to_file.items():
        print(f"  {k} -> {v}")

    # 2) Load class-wise distinctiveness per model and align classes
    per_model = []
    for mdl in MODEL_LABELS[:len(selected)]:
        per_model.append(load_classwise_json(model_to_file[mdl]))

    aligned, classes = align_classes(per_model)

    # 3) Normalize distinctiveness per class to weights
    distinct_w = normalize_distinctiveness_to_weights(aligned, classes)

    # 4) Load Optuna weights and align to classes
    optuna_w = load_optuna_weights(args.optuna_json, classes)

    # 5) Compute correlations per class
    results = {"Class": [], "Pearson": [], "Spearman": []}
    for cls in classes:
        x = np.array(distinct_w[cls], dtype=float)
        y = np.array(optuna_w[cls], dtype=float)
        # guard against constant arrays
        try:
            p, _ = pearsonr(x, y)
        except Exception:
            p = np.nan
        try:
            s, _ = spearmanr(x, y)
        except Exception:
            s = np.nan
        results["Class"].append(cls)
        results["Pearson"].append(round(p, 3) if p == p else p)    # keep NaN if any
        results["Spearman"].append(round(s, 3) if s == s else s)
        print(f"{cls}: Pearson={p:.3f}  Spearman={s:.3f}" if (p == p and s == s) else f"{cls}: Pearson={p}  Spearman={s}")

    corr_df = pd.DataFrame(results)
    corr_csv = os.path.join(args.outdir, "class_correlation_results.csv")
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved per-class correlations -> {corr_csv}")

    # 6) Plots
    plot_scatter_per_class(distinct_w, optuna_w, scatter_dir, baseline=args.baseline)
    heat_pearson = os.path.join(args.outdir, "correlation_heatmap_pearson.png")
    heat_spearman = os.path.join(args.outdir, "correlation_heatmap_spearman.png")
    plot_correlation_heatmap(distinct_w, optuna_w, heat_pearson, corr_type="pearson")
    plot_correlation_heatmap(distinct_w, optuna_w, heat_spearman, corr_type="spearman")
    print(f"Saved heatmaps -> {heat_pearson}  and  {heat_spearman}")

if __name__ == "__main__":
    main()