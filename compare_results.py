
"""
Script to compare two metrics JSON files.
Ignores AUROC. Computes differences in subset means (F1, Youden-index, Accuracy)
and per-class differences for specified diseases.

./compare_metrics.py metrics_old.json metrics_new.json
"""

import json
import argparse
import sys

# List of diseases to compare
DISEASES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

def load_metrics(path):
    """Load a metrics JSON file and return its contents as a dict."""
    try:
        with open(path, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error loading '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    return metrics

def get_value(dct, key, subkey=None):
    """
    Safely get a value from nested dictionaries.
    If subkey is provided, returns dct[key][subkey]. Otherwise returns dct[key].
    """
    if key not in dct:
        raise KeyError(f"Key '{key}' not found in metrics.")
    if subkey is None:
        return dct[key]
    if subkey not in dct[key]:
        raise KeyError(f"Subkey '{subkey}' not found under '{key}'.")
    return dct[key][subkey]

def main():
    parser = argparse.ArgumentParser(
        description="Compare two metrics.json files (ignoring AUROC). "
                    "Prints differences in subset means and per-class metrics for selected diseases."
    )
    parser.add_argument(
        'file1', metavar='metrics1.json',
        help="Path to the first metrics JSON file."
    )
    parser.add_argument(
        'file2', metavar='metrics2.json',
        help="Path to the second metrics JSON file."
    )
    args = parser.parse_args()

    # Load both metric files
    m1 = load_metrics(args.file1)
    m2 = load_metrics(args.file2)

    # Define subset-mean keys we care about
    subset_keys = [
        'F1_subset_mean',
        'Youden-index_subset_mean',
        'Accuracy_subset_mean'
    ]

    print("\n=== Subset-Mean Differences (file2 - file1) ===")
    for key in subset_keys:
        try:
            v1 = get_value(m1, key)
            v2 = get_value(m2, key)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        diff = v2 - v1
        print(f"{key}: {v2:.6f}  –  {v1:.6f}  =  {diff:.6f}")

    print("\n=== Per-Class Differences for Selected Diseases ===")
    # Metric dictionaries in each file
    f1_1 = m1.get('F1_per_class', {})
    f1_2 = m2.get('F1_per_class', {})
    youden_1 = m1.get('Youden_per_class', {})
    youden_2 = m2.get('Youden_per_class', {})
    acc_1 = m1.get('Accuracy_per_class', {})
    acc_2 = m2.get('Accuracy_per_class', {})

    for disease in DISEASES:
        print(f"\n-- {disease} --")
        # F1 difference
        try:
            f1_v1 = f1_1[disease]
            f1_v2 = f1_2[disease]
            f1_diff = f1_v2 - f1_v1
            print(f"F1_per_class: {f1_v2:.6f}  –  {f1_v1:.6f}  =  {f1_diff:.6f}")
        except KeyError:
            print(f"F1_per_class: Key '{disease}' not found in one of the files.")

        # Youden difference
        try:
            y_v1 = youden_1[disease]
            y_v2 = youden_2[disease]
            y_diff = y_v2 - y_v1
            print(f"Youden_per_class: {y_v2:.6f}  –  {y_v1:.6f}  =  {y_diff:.6f}")
        except KeyError:
            print(f"Youden_per_class: Key '{disease}' not found in one of the files.")

        # Accuracy difference
        try:
            a_v1 = acc_1[disease]
            a_v2 = acc_2[disease]
            a_diff = a_v2 - a_v1
            print(f"Accuracy_per_class: {a_v2:.6f}  –  {a_v1:.6f}  =  {a_diff:.6f}")
        except KeyError:
            print(f"Accuracy_per_class: Key '{disease}' not found in one of the files.")


if __name__ == "__main__":
    main()
