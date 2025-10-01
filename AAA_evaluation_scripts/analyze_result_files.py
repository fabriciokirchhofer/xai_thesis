import os, json
import pandas as pd

root_dir = "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results"

experiment_rows = []

for run_dir, dirs, files in os.walk(root_dir):
    if "metrics.json" not in files:
        continue

    # run_dir example: .../ensemble_DeepLift_320_no_baseline_29.09.2025/001_distinctiveness_voting_test_5m_20250929_114941
    run_folder = os.path.basename(run_dir)
    exp_dir = os.path.dirname(run_dir)
    exp_folder = os.path.basename(exp_dir)

    # ---- Parse parent experiment folder (saliency, size, baseline flag)
    # Examples:
    #   ensemble_DeepLift_320_no_baseline_29.09.2025
    #   ensemble_DeepLift_320_w_baseline_29.09.2025
    #   ensemble_GradCAM_original_24.09.2025
    exp_parts = exp_folder.split("_")
    # Guard: must start with 'ensemble'
    if not exp_parts or exp_parts[0].lower() != "ensemble":
        continue

    saliency = exp_parts[1] if len(exp_parts) > 1 else "Unknown"
    input_size = exp_parts[2] if len(exp_parts) > 2 else "Unknown"

    # Baseline flag (optional; not a required output column but useful context)
    baseline_flag = None
    if "w" in exp_parts and "baseline" in exp_parts:
        w_idx = exp_parts.index("w")
        # ensure next token is 'baseline'
        if w_idx + 1 < len(exp_parts) and exp_parts[w_idx + 1] == "baseline":
            baseline_flag = "with baseline"
    elif "no" in exp_parts and "baseline" in exp_parts:
        no_idx = exp_parts.index("no")
        if no_idx + 1 < len(exp_parts) and exp_parts[no_idx + 1] == "baseline":
            baseline_flag = "no baseline"

    # ---- Parse run folder (ensemble method + eval type)
    # Example run folder parts:
    #   ['001','distinctiveness','voting','test','5m','20250929','114941']
    rparts = run_folder.split("_")
    # Some runs may start with a numeric counter; skip it when present
    start_idx = 1 if rparts and rparts[0].isdigit() else 0

    method_word = rparts[start_idx] if len(rparts) > start_idx else "unknown"
    method_kind = rparts[start_idx + 1] if len(rparts) > start_idx + 1 else "unknown"
    eval_token = rparts[start_idx + 2] if len(rparts) > start_idx + 2 else "unknown"

    # Ensemble method name
    if method_word.lower() == "average":
        ensemble = "Average " + method_kind.capitalize()
    else:
        ensemble = method_word.capitalize() + " " + method_kind.capitalize()

    eval_map = {"val": "validation", "test": "test"}
    eval_type = eval_map.get(eval_token.lower(), eval_token.lower())

    # ---- Load metrics
    file_path = os.path.join(run_dir, "metrics.json")
    try:
        with open(file_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        # Skip malformed results rather than crash the aggregation
        continue

    experiment_rows.append({
        "Experiment": exp_folder,  # parent experiment identifier
        "Run folder": run_folder,  # optional, helps disambiguate multiple runs
        "Saliency method": saliency,
        "Input size": input_size,
        "Baseline": baseline_flag,
        "Ensemble method": ensemble,
        "Evaluation type": eval_type,
        "AUROC_subset_mean": metrics.get("AUROC_subset_mean"),
        "F1_subset_mean": metrics.get("F1_subset_mean"),
        "Youden-index_subset_mean": metrics.get("Youden-index_subset_mean"),
        "Accuracy_subset_mean": metrics.get("Accuracy_subset_mean"),
    })


# Create DataFrame
df = pd.DataFrame(experiment_rows)

# (Optional) Sort DataFrame for easier inspection: e.g., by Saliency, Input size, Ensemble, Eval type
saliency_order = ["LRP", "GradCAM", "DeepLift", "IG"]
ensemble_order = ["Distinctiveness Voting", "Distinctiveness Weighted", "Average Voting", "Average Weighted"]
eval_order = ["validation", "test"]
# Convert to categorical for ordering
df["Saliency method"] = pd.Categorical(df["Saliency method"], categories=saliency_order, ordered=True)
df["Ensemble method"] = pd.Categorical(df["Ensemble method"], categories=ensemble_order, ordered=True)
df["Evaluation type"] = pd.Categorical(df["Evaluation type"], categories=eval_order, ordered=True)
# Sort by saliency, input (numeric), ensemble, then eval type
df["input_num"] = df["Input size"].str.split("x").str[0]
df = df.sort_values(["Saliency method", "input_num", "Ensemble method", "Evaluation type"]).drop(columns="input_num")

# Preview the aggregated DataFrame
#print(df.to_string(index=False))

df_test = df[df["Evaluation type"] == "test"]
df_test = df.iloc[:, 2:]
print(df_test)
df_test_voting_F1 = df[df["Ensemble method"] == "Distinctiveness Voting"][df["Evaluation type"] == "test"][["Saliency method", "Input size", "Baseline", "F1_subset_mean"]]
df_test_weighted_F1 = df[df["Ensemble method"] == "Distinctiveness Weighted"][df["Evaluation type"] == "test"][["Saliency method", "Input size", "Baseline", "F1_subset_mean"]]
df_val = df[df["Evaluation type"] == "validation"]
print(df_test_weighted_F1)

# Save the full table (all experiments, all eval types)
# output_path = "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results_summary.csv"
# df.to_csv(output_path, index=False)

# print(f"Saved summary table to {output_path}")
