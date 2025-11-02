import os
import argparse
import numpy as np
import string
import csv

# Run with: python npy_to_csv_exporter.py --root /path/to/folder/with/npy/files

# Define class labels as in run_experiments.py
CLASS_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

def npy_to_csv(npy_file):
    """Load npy as dict (ensemble) or 2D array (per-model) and write corresponding CSV."""
    output_file = os.path.splitext(npy_file)[0] + ".csv"
    obj = np.load(npy_file, allow_pickle=True)
    # If dict (ensemble thresholds)
    if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
        thresholds = obj.item()
        row = [thresholds[k] for k in CLASS_LABELS]
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CLASS_LABELS)
            writer.writerow(row)
        print(f"Saved {output_file}")
        return
    # If 2D array (per model thresholds)
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        model_names = [f"Model_{letter}" for letter in string.ascii_uppercase[:obj.shape[0]]]
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["model"] + CLASS_LABELS)
            for model, row in zip(model_names, obj):
                writer.writerow([model] + list(row))
        print(f"Saved {output_file}")
        return
    # Fallthrough: Unexpected
    print(f"ERROR: {npy_file} is not a recognized dict or 2D array format.")

def find_and_convert(root_dir, filenames):
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            if fname in filenames:
                npy_to_csv(os.path.join(dirpath, fname))

def main():
    parser = argparse.ArgumentParser(description="Convert specified .npy files to .csv recursively.")
    parser.add_argument("--root", required=True, help="Root directory to search for .npy files.")
    args = parser.parse_args()
    find_and_convert(args.root, ["thresholds.npy", "per_model_voting_thresholds.npy"])

if __name__ == "__main__":
    main()
