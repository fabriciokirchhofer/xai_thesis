# xai_thesis

Thesis repository: Interpretability-based Model Ensembling Strategies for Medical Image Classification Models

## Overview

This repository implements ensemble strategies for medical image classification using explainable AI (XAI) methods. The project focuses on CheXpert dataset classification using multiple deep learning architectures (DenseNet121, ResNet152) and explores distinctiveness-based weighting for ensemble predictions.

## Installation

### Prerequisites

- Python 3.8
- CUDA-capable GPU (recommended)
- Conda or Miniconda

### Setup

1. Create conda environment from `requirements.yml`:
```bash
conda env create -f requirements.yml
conda activate xai
```

2. Install pretrained models in `third_party/pretrainedmodels/` (checkpoint files should be placed according to paths in `config.json`)

3. Update paths in `config.json`:
   - Adjust `device` to match your GPU setup
   - Update checkpoint paths to point to your model files
   - Update data paths to point to CheXpert dataset location

## Project Structure

```
xai_thesis/
├── run_experiments.py      # Main ensemble evaluation script
├── optimizer.py            # Optuna-based weight optimization
├── grid_search.py          # Hyperparameter grid search
├── config.json             # Configuration file (paths, models, ensemble settings)
├── third_party/
│   ├── run_models.py       # Individual model evaluation
│   ├── utils.py            # Utility functions (XAI, distinctiveness, metrics)
│   ├── models.py           # Model architectures
│   └── dataset.py          # Data loading
├── ensemble/               # Ensemble strategies and evaluation
└── AAA_evaluation_scripts/ # Analysis and visualization scripts
```

## Usage

### Main Ensemble Evaluation

Run ensemble experiments with configuration from `config.json`:

```bash
python run_experiments.py --config config.json
```

This script:
- Loads models specified in `config.json`
- Computes predictions on validation/test set
- Applies ensemble strategy (average, distinctiveness_weighted, voting, etc.)
- Evaluates metrics (AUROC, F1, Youden, Accuracy)
- Saves results to timestamped output directory

### Weight Optimization

Optimize ensemble weights using Optuna:

```bash
python optimizer.py --config config.json --output optimized_weights.json --trials 300
```

### Grid Search

Perform grid search over hyperparameters (a, b) for weight sharpening:

```bash
python grid_search.py
```

Note: Grid search parameters are configured in `config.json` under `grid_search` section.

### Individual Model Evaluation

Evaluate a single model:

```bash
cd third_party
python run_models.py --model DenseNet121 --ckpt path/to/checkpoint.ckpt
```

## Configuration

The `config.json` file contains:

- **models**: List of model configurations (architecture, checkpoint paths, batch size)
- **ensemble**: Ensemble strategy, distinctiveness files, threshold tuning settings
- **evaluation**: Metrics to compute, evaluation subset tasks
- **grid_search**: Grid search parameters (if enabled)
- **saliency_script**: XAI method settings (GradCAM, LRP, DeepLift, etc.)

**Important**: Update hardcoded paths in `config.json` and scripts to match your environment:
- Checkpoint paths
- CheXpert dataset paths (`/home/fkirchhofer/data/CheXpert-v1.0/`)
- Output directories

## Ensemble Strategies

- **average**: Simple averaging of model probabilities
- **average_voting**: Voting-based ensemble with per-model thresholds
- **distinctiveness_weighted**: Weight models by XAI-based distinctiveness scores
- **distinctiveness_voting**: Voting with distinctiveness-weighted votes

## Output

Results are saved in timestamped directories containing:
- `metrics.json`: Evaluation metrics
- `thresholds.npy`: Optimal thresholds per class
- `ensemble_probs.npy`: Ensemble probability predictions
- `plots/`: ROC curves, threshold effects, model analysis
- `config.json`: Copy of configuration for reproducibility

## Notes

- Hardcoded paths: Some scripts contain hardcoded paths (e.g., `/home/fkirchhofer/`). Update these to match your system.
- GPU device: Configured via `config.json` `device` field (default: `cuda:2`).
- Data paths: CheXpert dataset expected at paths specified in `third_party/run_models.py`
