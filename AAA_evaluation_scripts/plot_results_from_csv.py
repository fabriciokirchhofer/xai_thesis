import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Config ---
csv_path = "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results_summary.csv"
ensemble_method_main = "distinctiveness weighted"  # main method to compare against
baseline_ensemble_methods = ["average weighted"]
evaluation_set = "validation"  # "validation" or "test"

# --- Load ---
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

# --- Helper: safe string lower ---
def _lower(s):
    return s.astype(str).str.strip().str.lower()

# --- Filter main subset (Distinctiveness Weighted) ---
mask_main = (
    (_lower(df["Ensemble method"]) == ensemble_method_main) &
    (_lower(df["Evaluation type"]) == evaluation_set)
)
main_cols = ["Saliency method", "Input size", "Baseline", "F1_subset_mean", "Ensemble method"]
sub_main = df.loc[mask_main, main_cols].copy()

# --- Filter BASELINE experiments (no saliency computations) ---
# Heuristics: rows where ensemble method is in the baseline list AND saliency method is empty/none/baseline/NaN
baseline_mask = (_lower(df["Evaluation type"]) == evaluation_set) & (_lower(df["Ensemble method"]).isin([m.lower() for m in baseline_ensemble_methods]))
# identify rows without saliency
no_saliency_mask = (df["Saliency method"].isna() |_lower(df["Saliency method"]).isin(["", "none", "baseline", "no saliency", "na", "-", "--"]))
sub_base = df.loc[baseline_mask & no_saliency_mask, main_cols].copy()

# Normalize/label baseline rows
sub_base["Saliency method"] = "BASELINE"
sub_base["Input size"] = sub_base["Input size"].fillna("—")

# --- Combine ---
sub = pd.concat([sub_main, sub_base], ignore_index=True)

# --- Sort values ---
order_saliency = ["BASELINE", "LRP", "GradCAM", "DeepLift", "IG"]
sub["Saliency method"] = pd.Categorical(sub["Saliency method"], categories=order_saliency, ordered=True)
sub.sort_values(["Saliency method", "Input size", "Baseline"], inplace=True)

# --- Labels ---
sub["label"] = (
    sub["Saliency method"].astype(str) + " | " +
    sub["Input size"].astype(str) + " | " +
    sub["Baseline"].astype(str))

# --- Output dir ---
out_dir = "/home/fkirchhofer/repo/xai_thesis/AAA_evaluation_scripts/01_results_comparison"
os.makedirs(out_dir, exist_ok=True)

# -----------------
# Bar chart with zoomed-in y-axis + per-method colors + value labels
# -----------------
plt.figure(figsize=(16, 6))
x = np.arange(len(sub))

colors = {
    "BASELINE": "#7f7f7f",
    "LRP": "#1f77b4",
    "GradCAM": "#ff7f0e",
    "DeepLift": "#2ca02c",
    "IG": "#d62728"
}
bar_colors = [colors.get(m, "gray") for m in sub["Saliency method"]]

bars = plt.bar(x, sub["F1_subset_mean"].values, color=bar_colors)
plt.xticks(x, sub["label"].tolist(), rotation=60, ha="right")
plt.ylabel("F1 (subset mean)")
plt.title(f"F1 scores — {ensemble_method_main.title()} vs Baseline — {evaluation_set} set")

# Zoom into ROI
min_val = sub["F1_subset_mean"].min()
max_val = sub["F1_subset_mean"].max()
margin = max(0.002, 0.02 * (max_val - min_val if max_val > min_val else 1))
plt.ylim(min_val - margin, max_val + margin)

# Value labels (5 decimals)
for bar, val in zip(bars, sub["F1_subset_mean"].values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.5f}",
             ha="center", va="bottom", fontsize=8, rotation=90)

plt.tight_layout()
file_name = f"{ensemble_method_main.replace(' ', '_')}_vs_baseline_{evaluation_set}_set_bar.png"
plt.savefig(os.path.join(out_dir, file_name), dpi=200)
plt.close()

# -----------------
# Heatmap-style matrix (rows: Saliency|Input size|Ensemble method)
# -----------------
# mat = sub.pivot_table(
#     index=["Saliency method", "Input size", "Ensemble method"],
#     columns="Baseline",
#     values="F1_subset_mean",
#     aggfunc="mean"
# ).sort_index()

# fig, ax = plt.subplots(figsize=(10, max(4, 0.6*len(mat.index))))
# im = ax.imshow(mat.values, aspect="auto")
# ax.set_xticks(np.arange(len(mat.columns)))
# ax.set_xticklabels(mat.columns.tolist(), rotation=0)
# row_labels = [f"{i[0]} | {i[1]} | {i[2]}" for i in mat.index]
# ax.set_yticks(np.arange(len(mat.index)))
# ax.set_yticklabels(row_labels)
# ax.set_title(f"F1 matrix — {ensemble_method_main.title()} + Baselines — {evaluation_set}")
# ax.set_xlabel("Baseline")
# ax.set_ylabel("Saliency | Input size | Ensemble")

# # Annotate values (5 decimals)
# for i in range(mat.shape[0]):
#     for j in range(mat.shape[1]):
#         val = mat.values[i, j]
#         if not pd.isna(val):
#             ax.text(j, i, f"{val:.5f}", ha="center", va="center", fontsize=8)

# fig.tight_layout()
# file_name = f"heat_{ensemble_method_main.replace(' ', '_')}_{evaluation_set}_with_baselines.png"
# fig.savefig(os.path.join(out_dir, file_name), dpi=200)
# plt.close()

