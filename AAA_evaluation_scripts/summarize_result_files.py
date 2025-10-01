import os, json
import pandas as pd

# Scan both the distinctiveness experiments and the plain baselines
roots = [
    "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results",
    "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_BASELINE",
]

experiment_rows = []

for root_dir in roots:
    for run_dir, dirs, files in os.walk(root_dir):
        if "metrics.json" not in files:
            continue

        # Example paths:
        #  - .../ensemble_results/ensemble_DeepLift_320_no_baseline_29.09.2025/001_distinctiveness_voting_test_5m_20250929_114941
        #  - .../ensemble_BASELINE/001_average_voting_val_5m_20250922_104705
        run_folder = os.path.basename(run_dir)
        exp_dir = os.path.dirname(run_dir)
        exp_folder = os.path.basename(exp_dir)

        # ---------- Determine context (saliency, input size, baseline flag)
        saliency = "Unknown"
        input_size = "Unknown"
        baseline_flag = None
        is_baseline_bucket = (exp_folder == "ensemble_BASELINE")

        if is_baseline_bucket:
            # Baseline bucket has no saliency or size — keep schema consistent
            saliency = "BASELINE"
            input_size = "N/A"
        else:
            # Expect folders like: ensemble_DeepLift_320_no_baseline_29.09.2025
            exp_parts = exp_folder.split("_")
            # Guard: must start with 'ensemble'
            if not exp_parts or exp_parts[0].lower() != "ensemble":
                # Skip anything unexpected in ensemble_results
                continue

            saliency = exp_parts[1] if len(exp_parts) > 1 else "Unknown"
            input_size = exp_parts[2] if len(exp_parts) > 2 else "Unknown"

            # Optional baseline flag
            if "w" in exp_parts and "baseline" in exp_parts:
                w_idx = exp_parts.index("w")
                if w_idx + 1 < len(exp_parts) and exp_parts[w_idx + 1] == "baseline":
                    baseline_flag = "with baseline"
            elif "no" in exp_parts and "baseline" in exp_parts:
                no_idx = exp_parts.index("no")
                if no_idx + 1 < len(exp_parts) and exp_parts[no_idx + 1] == "baseline":
                    baseline_flag = "no baseline"

        # ---------- Parse run folder (ensemble method + eval type)
        # Patterns:
        #  distinctiveness_*_*_...      -> e.g., distinctiveness_voting_test_5m_...
        #  average_val_*                -> baseline weighted (average of probs) on validation
        #  average_test_*               -> baseline weighted on test
        #  average_voting_val_*         -> baseline voting on validation
        #  average_voting_test_*        -> baseline voting on test
        rparts = run_folder.split("_")
        start_idx = 1 if rparts and rparts[0].isdigit() else 0

        method_word = rparts[start_idx].lower() if len(rparts) > start_idx else "unknown"

        ensemble = "Unknown"
        eval_token = "unknown"

        if method_word == "distinctiveness":
            # Expect ... distinctiveness voting|weighted val|test ...
            method_kind = rparts[start_idx + 1].lower() if len(rparts) > start_idx + 1 else "unknown"
            eval_token = rparts[start_idx + 2].lower() if len(rparts) > start_idx + 2 else "unknown"
            ensemble = method_word.capitalize() + " " + method_kind.capitalize()

        elif method_word == "average":
            # Two formats:
            #   average voting val|test ...
            #   average val|test ...
            next_token = rparts[start_idx + 1].lower() if len(rparts) > start_idx + 1 else "unknown"
            if next_token == "voting":
                ensemble = "Average Voting"
                eval_token = rparts[start_idx + 2].lower() if len(rparts) > start_idx + 2 else "unknown"
            else:
                # No explicit "voting" => this is the probability-averaging ("weighted") baseline
                ensemble = "Average Weighted"
                eval_token = next_token

        else:
            # Fallback to previous generic behavior (in case of unforeseen names)
            method_kind = rparts[start_idx + 1] if len(rparts) > start_idx + 1 else "unknown"
            eval_token = rparts[start_idx + 2] if len(rparts) > start_idx + 2 else "unknown"
            if method_word == "average":
                ensemble = "Average " + method_kind.capitalize()
            else:
                ensemble = method_word.capitalize() + " " + method_kind.capitalize()

        eval_map = {"val": "validation", "test": "test"}
        eval_type = eval_map.get(eval_token, eval_token)

        # ---------- Load metrics
        file_path = os.path.join(run_dir, "metrics.json")
        try:
            with open(file_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            # Skip malformed results rather than crash the aggregation
            continue

        experiment_rows.append({
            "Experiment": exp_folder,   # parent experiment identifier
            "Run folder": run_folder,   # keeps multiple runs disambiguated
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

# ---------- Build DataFrame (preserve your existing ordering)
df = pd.DataFrame(experiment_rows)

# Category orders (extended to include BASELINE)
saliency_order = ["BASELINE", "LRP", "GradCAM", "DeepLift", "IG"]
ensemble_order = ["Distinctiveness Voting", "Distinctiveness Weighted", "Average Voting", "Average Weighted"]
eval_order = ["validation", "test"]

df["Saliency method"] = pd.Categorical(df["Saliency method"], categories=saliency_order, ordered=True)
df["Ensemble method"] = pd.Categorical(df["Ensemble method"], categories=ensemble_order, ordered=True)
df["Evaluation type"] = pd.Categorical(df["Evaluation type"], categories=eval_order, ordered=True)

# Sorting helper: pull a leading number from Input size if present (e.g., "320", "640"), else NaN
input_lead = (
    df["Input size"]
      .astype(str)
      .str.extract(r"^(\d+)", expand=False)
      .astype(float)
)
df = df.assign(_input_num=input_lead)
df = df.sort_values(["Saliency method", "_input_num", "Ensemble method", "Evaluation type"]).drop(columns="_input_num")

# --- Your previews (unchanged except they now include baselines where applicable)
df_test = df[df["Evaluation type"] == "test"]
df_test = df.iloc[:, 2:]
print(df_test)

df_test_voting_F1 = df[(df["Ensemble method"] == "Distinctiveness Voting") & (df["Evaluation type"] == "test")][
    ["Saliency method", "Input size", "Baseline", "F1_subset_mean"]
]
df_test_weighted_F1 = df[(df["Ensemble method"] == "Distinctiveness Weighted") & (df["Evaluation type"] == "test")][
    ["Saliency method", "Input size", "Baseline", "F1_subset_mean"]
]
df_val = df[df["Evaluation type"] == "validation"]
print(df_test_weighted_F1)

# Optional save:
output_path = "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results_summary.csv"
df.to_csv(output_path, index=False)
print(f"Saved summary table to {output_path}")






# import os, json
# import pandas as pd

# root_dir = "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results"

# experiment_rows = []

# for run_dir, dirs, files in os.walk(root_dir):
#     if "metrics.json" not in files:
#         continue

#     # run_dir example: .../ensemble_DeepLift_320_no_baseline_29.09.2025/001_distinctiveness_voting_test_5m_20250929_114941
#     run_folder = os.path.basename(run_dir)
#     exp_dir = os.path.dirname(run_dir)
#     exp_folder = os.path.basename(exp_dir)

#     # ---- Parse parent experiment folder (saliency, size, baseline flag)
#     # Examples:
#     #   ensemble_DeepLift_320_no_baseline_29.09.2025
#     #   ensemble_DeepLift_320_w_baseline_29.09.2025
#     #   ensemble_GradCAM_original_24.09.2025
#     exp_parts = exp_folder.split("_")
#     # Guard: must start with 'ensemble'
#     if not exp_parts or exp_parts[0].lower() != "ensemble":
#         continue

#     saliency = exp_parts[1] if len(exp_parts) > 1 else "Unknown"
#     input_size = exp_parts[2] if len(exp_parts) > 2 else "Unknown"

#     # Baseline flag (optional; not a required output column but useful context)
#     baseline_flag = None
#     if "w" in exp_parts and "baseline" in exp_parts:
#         w_idx = exp_parts.index("w")
#         # ensure next token is 'baseline'
#         if w_idx + 1 < len(exp_parts) and exp_parts[w_idx + 1] == "baseline":
#             baseline_flag = "with baseline"
#     elif "no" in exp_parts and "baseline" in exp_parts:
#         no_idx = exp_parts.index("no")
#         if no_idx + 1 < len(exp_parts) and exp_parts[no_idx + 1] == "baseline":
#             baseline_flag = "no baseline"

#     # ---- Parse run folder (ensemble method + eval type)
#     # Example run folder parts:
#     #   ['001','distinctiveness','voting','test','5m','20250929','114941']
#     rparts = run_folder.split("_")
#     # Some runs may start with a numeric counter; skip it when present
#     start_idx = 1 if rparts and rparts[0].isdigit() else 0

#     method_word = rparts[start_idx] if len(rparts) > start_idx else "unknown"
#     method_kind = rparts[start_idx + 1] if len(rparts) > start_idx + 1 else "unknown"
#     eval_token = rparts[start_idx + 2] if len(rparts) > start_idx + 2 else "unknown"

#     # Ensemble method name
#     if method_word.lower() == "average":
#         ensemble = "Average " + method_kind.capitalize()
#     else:
#         ensemble = method_word.capitalize() + " " + method_kind.capitalize()

#     eval_map = {"val": "validation", "test": "test"}
#     eval_type = eval_map.get(eval_token.lower(), eval_token.lower())

#     # ---- Load metrics
#     file_path = os.path.join(run_dir, "metrics.json")
#     try:
#         with open(file_path, "r") as f:
#             metrics = json.load(f)
#     except Exception:
#         # Skip malformed results rather than crash the aggregation
#         continue

#     experiment_rows.append({
#         "Experiment": exp_folder,  # parent experiment identifier
#         "Run folder": run_folder,  # optional, helps disambiguate multiple runs
#         "Saliency method": saliency,
#         "Input size": input_size,
#         "Baseline": baseline_flag,
#         "Ensemble method": ensemble,
#         "Evaluation type": eval_type,
#         "AUROC_subset_mean": metrics.get("AUROC_subset_mean"),
#         "F1_subset_mean": metrics.get("F1_subset_mean"),
#         "Youden-index_subset_mean": metrics.get("Youden-index_subset_mean"),
#         "Accuracy_subset_mean": metrics.get("Accuracy_subset_mean"),
#     })


# # Create DataFrame
# df = pd.DataFrame(experiment_rows)

# # (Optional) Sort DataFrame for easier inspection: e.g., by Saliency, Input size, Ensemble, Eval type
# saliency_order = ["LRP", "GradCAM", "DeepLift", "IG"]
# ensemble_order = ["Distinctiveness Voting", "Distinctiveness Weighted", "Average Voting", "Average Weighted"]
# eval_order = ["validation", "test"]
# # Convert to categorical for ordering
# df["Saliency method"] = pd.Categorical(df["Saliency method"], categories=saliency_order, ordered=True)
# df["Ensemble method"] = pd.Categorical(df["Ensemble method"], categories=ensemble_order, ordered=True)
# df["Evaluation type"] = pd.Categorical(df["Evaluation type"], categories=eval_order, ordered=True)
# # Sort by saliency, input (numeric), ensemble, then eval type
# df["input_num"] = df["Input size"].str.split("x").str[0]
# df = df.sort_values(["Saliency method", "input_num", "Ensemble method", "Evaluation type"]).drop(columns="input_num")

# # Preview the aggregated DataFrame
# #print(df.to_string(index=False))

# df_test = df[df["Evaluation type"] == "test"]
# df_test = df.iloc[:, 2:]
# print(df_test)
# df_test_voting_F1 = df[df["Ensemble method"] == "Distinctiveness Voting"][df["Evaluation type"] == "test"][["Saliency method", "Input size", "Baseline", "F1_subset_mean"]]
# df_test_weighted_F1 = df[df["Ensemble method"] == "Distinctiveness Weighted"][df["Evaluation type"] == "test"][["Saliency method", "Input size", "Baseline", "F1_subset_mean"]]
# df_val = df[df["Evaluation type"] == "validation"]
# print(df_test_weighted_F1)

# # Save the full table (all experiments, all eval types)
# # output_path = "/home/fkirchhofer/repo/xai_thesis/A_experiments_FINAL_01/ensemble_results_summary.csv"
# # df.to_csv(output_path, index=False)

# # print(f"Saved summary table to {output_path}")



