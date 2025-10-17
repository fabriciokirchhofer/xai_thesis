#!/usr/bin/env python3
"""
compare_ensembles.py

This script performs a bootstrap resampling analysis to compare F1-scores between ensemble strategies:
- Distinctiveness Weighted vs Average Weighted (DW vs AW)
- Distinctiveness Voting vs Average Voting (DV vs AV)

It computes F1 scores for each strategy on resampled test sets and uses bootstrapping to estimate the mean difference and 95% confidence interval of ΔF1. 
Additionally, a two-sided t-test is performed to assess significance, and violin plots of the F1 score distributions are generated.
"""
import argparse
import os
import numpy as np
# --- NumPy compatibility shim for libraries that still use np.float/np.int/np.bool ---
# Must run before importing seaborn (or any lib that references these aliases).
if not hasattr(np, "float"):
    np.float = float  # noqa: E305
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def find_ensemble_outputs(root: str):
    """Return file paths of ensemble probs and ensemble predictions (ensemble labels)."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("ensemble_probs.npy"):
                ens_probs_pth = os.path.join(dirpath, fn)
            if fn.endswith("ensemble_labels.npy"):
                ens_preds_pth = os.path.join(dirpath, fn)
    return ens_probs_pth, ens_preds_pth

def compute_f1(y_true, y_pred):
    """
    Compute F1 score for binary classification given true labels and predictions (0/1 arrays).
    Returns 0.0 if there are no true positives (to handle cases with zero_division).
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    # Calculate true positives, false positives, false negatives
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    if TP == 0:
        return 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def main():

    parser = argparse.ArgumentParser(description="Compare ensemble strategies via bootstrap F1 score differences.")

    # --- Input ensemble roots ---
    parser.add_argument("--DW_path", default="/home/fkirchhofer/repo/xai_thesis/ensemble_DeepLift_original_no_baseline_29.09.2025/001_distinctiveness_weighted_test_5m_20251016_152408",
                        help="Root to DW folder (shape (N, C))")
    
    parser.add_argument("--DV_path", default="/home/fkirchhofer/repo/xai_thesis/ensemble_DeepLift_original_no_baseline_29.09.2025/001_distinctiveness_voting_test_5m_20251016_155405",
                        help="Root to DV folder (shape (N, C))")
    
    # --- Output directory ---
    parser.add_argument("--output_dir", default="/home/fkirchhofer/repo/xai_thesis/AAA_evaluation_scripts/BOOT_DeepLift_test_original_no_baseline",
                    help="Directory to save output plots and summary table")
    
    # BASELINE - Should not change
    parser.add_argument("--AW_path", default="/home/fkirchhofer/repo/xai_thesis/ensemble_BASELINE_original_16.10.2025/001_average_test_5m_20251016_151924",
                        help="Root to AW folder (shape (N, C))")
    
    parser.add_argument("--AV_path", default="/home/fkirchhofer/repo/xai_thesis/ensemble_BASELINE_original_16.10.2025/001_average_voting_test_5m_20251016_160445",
                        help="root to AV folder (shape (N, C))")


    # GT - Should not change
    parser.add_argument("--gt_path", default="/home/fkirchhofer/repo/xai_thesis/ensemble_DeepLift_original_no_baseline_29.09.2025/001_distinctiveness_weighted_test_5m_20251016_152408/GT_labels.npy",
                        help="Path to ground_truth.npy (shape (N, C))")

    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap iterations (default: 1000)")
    args = parser.parse_args()

    AW_probs_path, AW_preds_path = find_ensemble_outputs(args.AW_path)
    AV_probs_path, AV_preds_path = find_ensemble_outputs(args.AV_path)
    DW_probs_path, DW_preds_path = find_ensemble_outputs(args.DW_path)
    DV_probs_path, DV_preds_path = find_ensemble_outputs(args.DV_path)

    # --- Helper to safely load arrays ---
    def load_array(path, name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] File not found for {name}: {path}")
        arr = np.load(path, allow_pickle=True)
        print(f"[INFO] Loaded {name}: shape={arr.shape}")
        return arr

    # --- Load all ensemble probability arrays ---
    DW_probs = load_array(DW_probs_path, "DW_probs")
    DV_probs = load_array(DV_probs_path, "DV_probs")
    AW_probs = load_array(AW_probs_path, "AW_probs")
    AV_probs = load_array(AV_probs_path, "AV_probs")

    # Stack into single array of shape (N, C, 4)
    ensemble_probs = np.stack([DW_probs, DV_probs, AW_probs, AV_probs], axis=-1)
    print(f"[INFO] Stacked ensemble_probs shape: {ensemble_probs.shape}")

    # --- Load all ensemble prediction arrays ---
    DW_preds = load_array(DW_preds_path, "DW_preds")
    DV_preds = load_array(DV_preds_path, "DV_preds")
    AW_preds = load_array(AW_preds_path, "AW_preds")
    AV_preds = load_array(AV_preds_path, "AV_preds")

    # Stack into single array of shape (N, C, 4)
    ensemble_preds = np.stack([AW_preds, DW_preds, AV_preds, DV_preds], axis=-1)
    print(f"[INFO] Stacked ensemble_preds shape: {ensemble_preds.shape}")

    # --- Load ground truth ---
    ground_truth = load_array(args.gt_path, "ground_truth")

    # (optional sanity check)
    assert ensemble_probs.shape[:2] == ground_truth.shape, \
        f"Shape mismatch: probs/preds {ensemble_probs.shape[:2]} vs GT {ground_truth.shape}"

    # If data was saved as Python objects (dtype=object), convert to numeric numpy arrays
    if hasattr(ensemble_preds, "dtype") and ensemble_preds.dtype == object:
        ensemble_preds = np.array([np.array(x) for x in ensemble_preds])
    if hasattr(ensemble_probs, "dtype") and ensemble_probs.dtype == object:
        ensemble_probs = np.array([np.array(x) for x in ensemble_probs])
    if hasattr(ground_truth, "dtype") and ground_truth.dtype == object:
        ground_truth = np.array([np.array(x) for x in ground_truth])

    # Ensure ensemble_preds has shape (N, C, 4). If the 4 strategies dimension is not the last axis, transpose it.
    if ensemble_preds.ndim == 3:
        strat_dim = None
        for dim in range(ensemble_preds.ndim):
            if ensemble_preds.shape[dim] == 4:
                strat_dim = dim
                break
        if strat_dim is None:
            raise ValueError("Prediction array does not have a dimension of size 4 (expected strategies axis).")
        if strat_dim != 2:
            # Move the strategy axis to the last position
            axes = list(range(ensemble_preds.ndim))
            axes.pop(strat_dim)
            axes.append(strat_dim)
            ensemble_preds = ensemble_preds.transpose(tuple(axes))
            ensemble_probs = ensemble_probs.transpose(tuple(axes))
    else:
        raise ValueError("Predictions array is not 3-dimensional as expected.")

    # Ensure ground_truth has shape (N, C)
    if ground_truth.ndim == 1:
        # If it's a flat array of length N*C, reshape to (N, C) (14 classes known in project)
        if ground_truth.shape[0] % 14 == 0:
            ground_truth = ground_truth.reshape(-1, 14)
    if ground_truth.ndim != 2:
        raise ValueError("Ground truth array is not 2-dimensional (N, C).")

    # Define class names (14 classes fixed in the project)
    class_names = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices"]
    
    # Focus on the 5 classes for final evaluation
    classes_of_interest = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    class_indices = [class_names.index(cls) for cls in classes_of_interest]

    # Set up random number generator for bootstrap - set seed here 
    rng = np.random.RandomState(0)
    n_samples = ground_truth.shape[0]

    # Bootstrap resampling of test data to estimate the distribution of F1 scores and differences (non-parametric approach with minimal assumptions).
    # Each bootstrap draw recomputes F1 for both strategies on a resampled set (paired comparison).
    # This yields a distribution of ΔF1 across resamples, from which we derive confidence intervals.
    # Dictionaries to store bootstrap difference distributions for each class
    dw_minus_aw_diffs = {}  # ΔF1 = F1(DW) – F1(AW)
    dv_minus_av_diffs = {}  # ΔF1 = F1(DV) – F1(AV)
    # Additionally, store bootstrap F1 score distributions for each strategy
    f1_AW_scores = {}
    f1_DW_scores = {}
    f1_AV_scores = {}
    f1_DV_scores = {}

    for cls_idx in class_indices:
        y_true_full = ground_truth[:, cls_idx].astype(int)
        # Extract binary predictions for this class from each strategy (last axis: 0=AW,1=DW,2=AV,3=DV)
        y_pred_AW = ensemble_preds[:, cls_idx, 0].astype(int)
        y_pred_DW = ensemble_preds[:, cls_idx, 1].astype(int)
        y_pred_AV = ensemble_preds[:, cls_idx, 2].astype(int)
        y_pred_DV = ensemble_preds[:, cls_idx, 3].astype(int)

        diffs_dw_aw = []
        diffs_dv_av = []
        f1_AW_vals = []
        f1_DW_vals = []
        f1_AV_vals = []
        f1_DV_vals = []

        # Bootstrap iterations
        for b_iteration in range(args.n_bootstraps):
            # Sample N indices with replacement
            idx = rng.randint(0, n_samples, size=n_samples)
            # Compute F1 for each strategy on the resampled set
            f1_AW = compute_f1(y_true_full[idx], y_pred_AW[idx])
            f1_DW = compute_f1(y_true_full[idx], y_pred_DW[idx])
            f1_AV = compute_f1(y_true_full[idx], y_pred_AV[idx])
            f1_DV = compute_f1(y_true_full[idx], y_pred_DV[idx])
            # Record F1 scores for each strategy
            f1_AW_vals.append(f1_AW)
            f1_DW_vals.append(f1_DW)
            f1_AV_vals.append(f1_AV)
            f1_DV_vals.append(f1_DV)
            # Record differences
            diffs_dw_aw.append(f1_DW - f1_AW)
            diffs_dv_av.append(f1_DV - f1_AV)
        # Store distributions for this class
        dw_minus_aw_diffs[cls_idx] = np.array(diffs_dw_aw)
        dv_minus_av_diffs[cls_idx] = np.array(diffs_dv_av)
        f1_AW_scores[cls_idx] = np.array(f1_AW_vals)
        f1_DW_scores[cls_idx] = np.array(f1_DW_vals)
        f1_AV_scores[cls_idx] = np.array(f1_AV_vals)
        f1_DV_scores[cls_idx] = np.array(f1_DV_vals)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Plot violin charts for absolute F1 score distributions
    # (visualize performance spread for each strategy and class)

    # ---- NEW: violinplots of ΔF1 per class (relative differences) ----
    import pandas as pd

    # DW – AW
    dw_aw_data = [np.asarray(dw_minus_aw_diffs[idx], dtype=float) for idx in class_indices]
    dw_aw_df = pd.DataFrame({
        "Class": np.repeat(classes_of_interest, [len(x) for x in dw_aw_data]),
        "ΔF1": np.concatenate(dw_aw_data),
        "Comparison": "DW – AW"
    })
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Class", y="ΔF1", data=dw_aw_df, cut=0)
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("F1 Score Difference (DW – AW)")
    plt.title("Bootstrap ΔF1 by Class: Distinctiveness Weighted vs Average Weighted")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "violin_DeltaF1_DW_vs_AW_by_class.png"))
    plt.close()

    # DV – AV
    dv_av_data = [np.asarray(dv_minus_av_diffs[idx], dtype=float) for idx in class_indices]
    dv_av_df = pd.DataFrame({
        "Class": np.repeat(classes_of_interest, [len(x) for x in dv_av_data]),
        "ΔF1": np.concatenate(dv_av_data),
        "Comparison": "DV – AV"
    })
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Class", y="ΔF1", data=dv_av_df, cut=0)
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("F1 Score Difference (DV – AV)")
    plt.title("Bootstrap ΔF1 by Class: Distinctiveness Voting vs Average Voting")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "violin_DeltaF1_DV_vs_AV_by_class.png"))
    plt.close()









    # 1. AW vs DW (Weighted strategies)
    weighted_classes = []
    weighted_strategies = []
    weighted_scores = []
    for cls_idx in class_indices:
        cls_name = class_names[cls_idx]
        scores_AW = f1_AW_scores[cls_idx]
        scores_DW = f1_DW_scores[cls_idx]
        weighted_classes.extend([cls_name] * len(scores_AW) + [cls_name] * len(scores_DW))
        weighted_strategies.extend(["AW"] * len(scores_AW) + ["DW"] * len(scores_DW))
        weighted_scores.extend(scores_AW.tolist() + scores_DW.tolist())
    weighted_df = pd.DataFrame({"Class": weighted_classes, "Strategy": weighted_strategies, "F1": weighted_scores})
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Class", y="F1", hue="Strategy", data=weighted_df, cut=0)
    plt.ylabel("F1 Score")
    plt.title("Bootstrap F1 Scores: Distinctiveness Weighted vs Average Weighted")
    #plt.ylim(0.4, 0.7)  # focus y-axis on observed F1 range (~0.57 ± some)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "violin_F1_DW_vs_AW.png"))
    plt.close()
    # 2. AV vs DV (Voting strategies)
    voting_classes = []
    voting_strategies = []
    voting_scores = []
    for cls_idx in class_indices:
        cls_name = class_names[cls_idx]
        scores_AV = f1_AV_scores[cls_idx]
        scores_DV = f1_DV_scores[cls_idx]
        voting_classes.extend([cls_name] * len(scores_AV) + [cls_name] * len(scores_DV))
        voting_strategies.extend(["AV"] * len(scores_AV) + ["DV"] * len(scores_DV))
        voting_scores.extend(scores_AV.tolist() + scores_DV.tolist())
    voting_df = pd.DataFrame({"Class": voting_classes, "Strategy": voting_strategies, "F1": voting_scores})
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Class", y="F1", hue="Strategy", data=voting_df, cut=0)
    plt.ylabel("F1 Score")
    plt.title("Bootstrap F1 Scores: Distinctiveness Voting vs Average Voting")
    #plt.ylim(0.4, 0.7)  # focus y-axis on observed F1 range (~0.57 ± some)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "violin_F1_DV_vs_AV.png"))
    plt.close()

    # Perform a two-sided one-sample t-test on ΔF1 to check if mean difference ≠ 0 (null: no difference).
    # The paired t-test is a common approach for comparing model performance:contentReference[oaicite:3]{index=3} and provides a p-value indicating significance.
    # A low p-value (e.g. < 0.05) suggests the performance difference is unlikely due to random chance:contentReference[oaicite:4]{index=4}.
    # We also report Cohen's d (effect size) and the 95% confidence interval for ΔF1 (from bootstrap percentiles).
    summary_rows = []
    for cls_idx in class_indices:
        diffs1 = dw_minus_aw_diffs[cls_idx]
        diffs2 = dv_minus_av_diffs[cls_idx]

        # Two-sided one-sample t-test against 0 for ΔF1 (DW–AW)
        t1, p_t1 = stats.ttest_1samp(diffs1, popmean=0.0, alternative="two-sided")
        # Cohen's d (one-sample): mean/SD of ΔF1
        d1 = np.mean(diffs1) / (np.std(diffs1, ddof=1) + 1e-12)

        # Two-sided one-sample t-test against 0 for ΔF1 (DV–AV)
        t2, p_t2 = stats.ttest_1samp(diffs2, popmean=0.0, alternative="two-sided")
        d2 = np.mean(diffs2) / (np.std(diffs2, ddof=1) + 1e-12)

        # Calculate mean and 95% CI bounds from bootstrap percentiles
        mean1 = np.mean(diffs1)
        ci_low1, ci_high1 = np.percentile(diffs1, [2.5, 97.5])
        mean2 = np.mean(diffs2)
        ci_low2, ci_high2 = np.percentile(diffs2, [2.5, 97.5])

        # Format mean ± CI for output table
        mean_ci_1 = f"{mean1:.4f} ({ci_low1:.4f}–{ci_high1:.4f})"
        mean_ci_2 = f"{mean2:.4f} ({ci_low2:.4f}–{ci_high2:.4f})"

        summary_rows.append({
        "Class": class_names[cls_idx],
        "ΔF1 DW–AW (mean ± 95% CI)": mean_ci_1,
        "t (DW–AW)": f"{t1:.3f}",
        "p (DW–AW, two-sided t-test)": "<0.001" if p_t1 < 1e-3 else f"{p_t1:.3f}",
        "Cohen d (DW–AW)": f"{d1:.3f}",
        "ΔF1 DV–AV (mean ± 95% CI)": mean_ci_2,
        "t (DV–AV)": f"{t2:.3f}",
        "p (DV–AV, two-sided t-test)": "<0.001" if p_t2 < 1e-3 else f"{p_t2:.3f}",
        "Cohen d (DV–AV)": f"{d2:.3f}"
    })




    # ---- NEW: overall (macro-F1 across classes) paired-bootstrap analysis ----
    # We reuse ensemble_preds of shape (N, C, 4) and ground_truth of shape (N, C).
    # Strategy indices: 0=AW, 1=DW, 2=AV, 3=DV
    pred_AW = ensemble_preds[:, :, 0].astype(int)
    pred_DW = ensemble_preds[:, :, 1].astype(int)
    pred_AV = ensemble_preds[:, :, 2].astype(int)
    pred_DV = ensemble_preds[:, :, 3].astype(int)

    macroF1_AW, macroF1_DW, macroF1_AV, macroF1_DV = [], [], [], []
    delta_macro_DW_AW, delta_macro_DV_AV = [], []

    # Use a single resample per iteration for ALL classes (paired comparison at the strategy level)
    rng_overall = np.random.RandomState(123)
    for b in range(args.n_bootstraps):
        idx = rng_overall.randint(0, n_samples, size=n_samples)
        f1_aw_list, f1_dw_list, f1_av_list, f1_dv_list = [], [], [], []
        for cls_idx in class_indices:
            y_true_b = ground_truth[idx, cls_idx].astype(int)
            f1_aw_list.append(compute_f1(y_true_b, pred_AW[idx, cls_idx]))
            f1_dw_list.append(compute_f1(y_true_b, pred_DW[idx, cls_idx]))
            f1_av_list.append(compute_f1(y_true_b, pred_AV[idx, cls_idx]))
            f1_dv_list.append(compute_f1(y_true_b, pred_DV[idx, cls_idx]))
        # macro-F1 across the five classes
        m_aw = float(np.mean(f1_aw_list))
        m_dw = float(np.mean(f1_dw_list))
        m_av = float(np.mean(f1_av_list))
        m_dv = float(np.mean(f1_dv_list))
        macroF1_AW.append(m_aw); macroF1_DW.append(m_dw)
        macroF1_AV.append(m_av); macroF1_DV.append(m_dv)
        delta_macro_DW_AW.append(m_dw - m_aw)
        delta_macro_DV_AV.append(m_dv - m_av)

    macroF1_AW = np.array(macroF1_AW)
    macroF1_DW = np.array(macroF1_DW)
    macroF1_AV = np.array(macroF1_AV)
    macroF1_DV = np.array(macroF1_DV)
    delta_macro_DW_AW = np.array(delta_macro_DW_AW)
    delta_macro_DV_AV = np.array(delta_macro_DV_AV)

    # ---- Plots: absolute macro-F1 per strategy (zoomed y-scale) ----
    import pandas as pd
    overall_df_w = pd.DataFrame({
        "Strategy": ["AW"] * len(macroF1_AW) + ["DW"] * len(macroF1_DW),
        "MacroF1": np.concatenate([macroF1_AW, macroF1_DW]),
    })
    plt.figure(figsize=(6.5, 4.5))
    sns.violinplot(x="Strategy", y="MacroF1", data=overall_df_w, cut=0)
    plt.title("Bootstrap Macro-F1 (5 classes): Distinctiveness Weighted vs Average Weighted")
    plt.ylim(0.4, 0.7)  # adjust if your scores live elsewhere
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "overall_violin_macroF1_DW_vs_AW.png"))
    plt.close()

    overall_df_v = pd.DataFrame({
        "Strategy": ["AV"] * len(macroF1_AV) + ["DV"] * len(macroF1_DV),
        "MacroF1": np.concatenate([macroF1_AV, macroF1_DV]),
    })
    plt.figure(figsize=(6.5, 4.5))
    sns.violinplot(x="Strategy", y="MacroF1", data=overall_df_v, cut=0)
    plt.title("Bootstrap Macro-F1 (5 classes): Distinctiveness Voting vs Average Voting")
    plt.ylim(0.4, 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "overall_violin_macroF1_DV_vs_AV.png"))
    plt.close()

    # ---- Plots: Δ macro-F1 distributions (DW–AW and DV–AV) ----
    delta_overall_df = pd.DataFrame({
        "Δ MacroF1": np.concatenate([delta_macro_DW_AW, delta_macro_DV_AV]),
        "Comparison": (["DW – AW"] * len(delta_macro_DW_AW)) + (["DV – AV"] * len(delta_macro_DV_AV))
    })
    plt.figure(figsize=(7.5, 4.5))
    sns.violinplot(x="Comparison", y="Δ MacroF1", data=delta_overall_df, cut=0)
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Bootstrap Δ Macro-F1 Distributions (Overall Across Classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "overall_violin_Delta_macroF1.png"))
    plt.close()

    # ---- Overall p-values (two-sided t-tests on Δ macro-F1), effect sizes, CIs ----
    t_dw_aw, p_dw_aw = stats.ttest_1samp(delta_macro_DW_AW, 0.0, alternative="two-sided")
    t_dv_av, p_dv_av = stats.ttest_1samp(delta_macro_DV_AV, 0.0, alternative="two-sided")
    d_dw_aw = np.mean(delta_macro_DW_AW) / (np.std(delta_macro_DW_AW, ddof=1) + 1e-12)
    d_dv_av = np.mean(delta_macro_DV_AV) / (np.std(delta_macro_DV_AV, ddof=1) + 1e-12)
    ci_dw_aw = np.percentile(delta_macro_DW_AW, [2.5, 97.5])
    ci_dv_av = np.percentile(delta_macro_DV_AV, [2.5, 97.5])

    overall_summary = pd.DataFrame([
        {
            "Comparison": "DW – AW (macro-F1)",
            "Δ mean": f"{np.mean(delta_macro_DW_AW):.4f}",
            "95% CI": f"{ci_dw_aw[0]:.4f} – {ci_dw_aw[1]:.4f}",
            "t": f"{t_dw_aw:.3f}",
            "p (two-sided t-test)": "<0.001" if p_dw_aw < 1e-3 else f"{p_dw_aw:.3f}",
            "Cohen d": f"{d_dw_aw:.3f}",
        },
        {
            "Comparison": "DV – AV (macro-F1)",
            "Δ mean": f"{np.mean(delta_macro_DV_AV):.4f}",
            "95% CI": f"{ci_dv_av[0]:.4f} – {ci_dv_av[1]:.4f}",
            "t": f"{t_dv_av:.3f}",
            "p (two-sided t-test)": "<0.001" if p_dv_av < 1e-3 else f"{p_dv_av:.3f}",
            "Cohen d": f"{d_dv_av:.3f}",
        },
    ])
    overall_summary.to_csv(os.path.join(args.output_dir, "overall_macroF1_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "overall_macroF1_summary.md"), "w") as f:
        f.write(overall_summary.to_markdown(index=False))
    print("\nOverall (macro-F1 across classes) summary:")
    print(overall_summary.to_string(index=False))





    results_df = pd.DataFrame(summary_rows)
    # Save the summary table as CSV
    results_df.to_csv(os.path.join(args.output_dir, "bootstrap_comparison_summary.csv"), index=False)
    # Also save as Markdown (optional, for easy viewing)
    with open(os.path.join(args.output_dir, "bootstrap_comparison_summary.md"), "w") as f:
        f.write(results_df.to_markdown(index=False))
    # Print summary table to console
    print("Bootstrap F1-score comparison summary (DW vs AW and DV vs AV):")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()


