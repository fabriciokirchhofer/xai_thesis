#!/usr/bin/env python3
"""
Generate and save a radar (spider) chart comparing multiple ensemble strategies.

How to use:
1. Edit the `json_files` list to include paths to your metrics.json files (2–5 files).
2. Edit the `model_names` list to include custom labels for each model (must match number of files).
3. Optionally change `output_png` for the desired output filename.
4. Run the script: python radar_plot.py
"""

import json
import math
import matplotlib.pyplot as plt
import datetime

# === USER CONFIGURATION ===
# List of paths to your metrics.json files (between 2 and 5 entries):
json_files = [
    "/home/fkirchhofer/repo/xai_thesis/ensemble_results_GradCAM/ensemble_results_average_f1_BASELINE_v1/baseline_validation_thresholds_5m_20250602_091402/metrics.json",
    "/home/fkirchhofer/repo/xai_thesis/ensemble_results_GradCAM/ensemble_results_distinctiveness_voting_f1/001_val_5m_20250707_113231/metrics.json",
    "/home/fkirchhofer/repo/xai_thesis/ensemble_results_GradCAM/ensemble_results_distinctiveness_weighted_f1/validation_thresholds_5m_20250602_093452/metrics.json"
]

# Custom names for each ensemble, in the same order as json_files:
model_names = [
    "Average (baseline)",
    "Distinctiveness voting",
    "Distinctiveness weighted"
    # add names corresponding to any additional files above
]

local_time_zone = datetime.timezone(datetime.timedelta(hours=2), name="CEST")
timestamp = datetime.datetime.now(local_time_zone).strftime("%Y%m%d_%H%M%S")
# Output image filename (PNG):
output_png = f"001_radar_comparison_{timestamp}.png"

# Validate inputs
if not (2 <= len(json_files) <= 5):
    raise ValueError("Please specify between 2 and 5 JSON files in `json_files`.")
if len(model_names) != len(json_files):
    raise ValueError("`model_names` length must match number of `json_files`.")

# Metrics to plot
metrics_keys = [
    "AUROC_subset_mean",
    "F1_subset_mean",
    "Youden-index_subset_mean",
    "Accuracy_subset_mean",
]

# Load values from each JSON
all_values = []  # list of lists of metric values
for path in json_files:
    with open(path, 'r') as f:
        data = json.load(f)
    values = [data[key] for key in metrics_keys]
    all_values.append(values)

# Number of metrics (axes)
N = len(metrics_keys)

# Compute angle for each axis in the plot (in radians)
angles = [i * 2 * math.pi / N for i in range(N)]
# Append the first angle again at end to close the loop
angles += angles[:1]

# Create the radar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Rotate so the first axis is at the top and go clockwise
ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)

# Draw metric labels on the axes
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_keys, fontsize=9)

# Set radial limits and ticks
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
ax.set_rlabel_position(30)  # move radial labels away from top

# Define up to 5 distinct colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot each ensemble's metrics
for idx, values in enumerate(all_values):
    # Close the loop by appending the first value again
    vals = values + [values[0]]
    ax.plot(angles, vals, color=colors[idx], linewidth=1, label=model_names[idx])
    # Optional: draw markers
    # ax.scatter(angles, vals, color=colors[idx], s=40)

# Add legend outside the plot
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

# Save to file
plt.tight_layout()
plt.savefig(output_png, dpi=300)