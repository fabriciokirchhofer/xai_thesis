#!/usr/bin/env python3
"""
optimize_ensemble_weights.py

This script finds an optimal weight matrix for ensembling multiple models on the CheXpert dataset,
and optionally tunes decision thresholds either per‑model (pre‑ensemble) or on the ensemble output (post‑ensemble),
maximizing the average F1-score for specified evaluation classes on the validation set.

Visualize on Optuna: optuna-dashboard sqlite:///optuna_ensemble.db

Usage:
    python optimize_ensemble_weights.py \
        --config path/to/config.json \
        --output path/to/weights_and_thresholds.json \
        [--no-normalize] \
        [--trials 50]
"""
import json
import os
import sys
import numpy as np
import torch
import optuna

from third_party import run_models, utils
from ensemble import evaluator, model_classes

optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimize ensemble weight matrix (and thresholds)."
    )
    parser.add_argument(
        "--config", type=str, default="/home/fkirchhofer/repo/xai_thesis/config.json",
        help="Path to JSON config file with model and ensemble settings."
    )
    parser.add_argument(
        "--output", type=str, default= "/home/fkirchhofer/repo/xai_thesis/optimized_weights_with_p.json",
        help="Path to save the optimized weights+thresholds JSON."
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="If set, do not normalize weights per class (otherwise weights sum to 1)."
    )
    parser.add_argument(
        "--trials", type=int, default=300,
        help="Number of Optuna trials per class (default: 30)."
    )
    args = parser.parse_args()

    # ----- load config -----
    with open(args.config, "r") as f:
        config = json.load(f)
    ensemble_cfg = config.get("ensemble", {})
    eval_cfg = config.get("evaluation", {})
    tune_cfg = ensemble_cfg.get("threshold_tuning", {})
    thresholds_path = ensemble_cfg.get("thresholds_path", None)

    # ----- define tasks -----
    # full CheXpert 14 tasks order
    tasks = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema",
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    # evaluation classes (override via config if provided)
    eval_tasks = config.get(
        "evaluation_sub_tasks",
        ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    )
    num_classes = len(eval_tasks)
    num_models = len(config["models"])
    model_names = [m["name"] for m in config["models"]]

    # ----- prepare models -----
    # hack to get default args without interference
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    base_args = run_models.parse_arguments()
    sys.argv = saved_argv

    models = []
    for m_cfg in config["models"]:
        m_args = utils.copy.deepcopy(base_args) if hasattr(utils, "copy") else __import__("copy").deepcopy(base_args)
        m_args.model = m_cfg.get("architecture", m_args.model)
        m_args.ckpt = m_cfg["checkpoint_path"]
        for k, v in m_cfg.get("overrides", {}).items():
            setattr(m_args, k, v)
        Wrapper = model_classes.get_model_wrapper(m_args.model)
        models.append(Wrapper(tasks=tasks, model_args=m_args))

    # ----- run models to get probabilities + true labels -----
    use_test = eval_cfg.get("evaluate_test_set", False)
    model_probs = []  # will be list of (N_samples x 14) arrays
    all_gt_labels = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, model in enumerate(models):
        # prepare loader
        data_loader = model.prepare_data_loader(
            default_data_conditions=False,
            batch_size_override=getattr(model.model_args, "batch_size", 16),
            test_set=use_test,
            assign=True
        )
        # collect GT once
        if all_gt_labels is None:
            all_lbls = []
            for _, batch_lbl in data_loader:
                all_lbls.append(batch_lbl.numpy())
            all_gt_labels = np.vstack(all_lbls)

        # run model
        logits = model.run_class_model().to(device)
        if eval_cfg.get("use_logits", False):
            probs = logits.detach().cpu().numpy()
            print(f"Model {model_names[i]}: using raw logits")
        else:
            raw = torch.sigmoid(logits).detach().cpu()
            # reduce multiple views per patient
            probs, gt_labels = utils.get_max_prob_per_view(
                probs=raw,
                gt_labels=all_gt_labels,
                tasks=tasks,
                args=model.model_args
            )
        model_probs.append(probs)
        print(f"Collected probs for model {model_names[i]} (shape {probs.shape})")

    # stack to shape (M, N, 14)
    model_probs = np.stack(model_probs, axis=0)
    N = model_probs.shape[1]
    print(f"Stacked model_probs: {model_probs.shape}")

    # indices of eval_tasks within the full tasks list
    eval_indices = [tasks.index(c) for c in eval_tasks]

    # ----- PRE‑ensemble threshold tuning per model -----
    thresholds_by_model = None
    if thresholds_path and os.path.exists(thresholds_path):
        loaded = np.load(thresholds_path, allow_pickle=True).item()
        thresholds_by_model = []
        for m_name in model_names:
            thr_dict = loaded.get(m_name, {})
            thresholds_by_model.append([thr_dict.get(c, 0.5) for c in eval_tasks])
        print("Loaded per‑model thresholds from file:", thresholds_by_model)

    elif tune_cfg.get("stage") == "pre":
        print(">>> performing PRE‑ensemble threshold tuning per model")
        thresholds_by_model = []
        for m in range(num_models):
            p_m = model_probs[m][:, eval_indices]       # (N x num_classes)
            y_m = gt_labels[:, eval_indices]
            thr_dict, _ = evaluator.find_optimal_thresholds(
                probabilities=p_m,
                ground_truth=y_m,
                tasks=eval_tasks,
                metric=tune_cfg.get("metric", "f1")
            )
            vec = [thr_dict[c] for c in eval_tasks]
            thresholds_by_model.append(vec)
            print(f" Model {model_names[m]} thresholds:", vec)

    else:
        print("No pre‑ensemble threshold tuning; using default 0.5 everywhere.")

    # ----- set up for post‑ensemble threshold tuning if requested -----
    do_post_thr = (tune_cfg.get("stage") == "post")
    ensemble_thresholds = {} if do_post_thr else None

    # ----- optimize weights per class -----
    weight_matrix = np.zeros((num_models, num_classes), dtype=float)
    output = {"F1_scores": {}}
    output["Thresholds"] = {}

    optuna_db = ensemble_cfg.get("optuna_db_path", "sqlite:///optuna_ensemble.db")
    for idx, cls in enumerate(eval_tasks):
        print(f"\n>>> Optimizing weights for class {cls}")
        def objective(trial):
            raw_w = [
                trial.suggest_float(f"w{m}_cls{idx}", 0.0, 1.0)
                for m in range(num_models)
            ]
            if args.no_normalize:
                w = np.array(raw_w, dtype=float)
            else:
                s = sum(raw_w)
                w = (np.array(raw_w)/s) if s > 0 else np.ones(num_models)/num_models

            # build binary preds
            if thresholds_by_model is not None:
                # threshold each model's probs pre‑ensemble
                bits = np.stack([
                    (model_probs[m][:, eval_indices[idx]] >= thresholds_by_model[m][idx]).astype(float)
                    for m in range(num_models)
                ], axis=0)  # shape (M, N)
                agg = np.dot(w, bits)  # shape (N,)
                preds = (agg >= 0.5).astype(int)
            else:
                # directly average probs
                pm = model_probs[:, :, eval_indices[idx]]  # (M, N)
                agg = np.dot(w, pm)
                agg = np.clip(agg, 0.0, 1.0)
                preds = (agg >= 0.5).astype(int)

            y_true = gt_labels[:, eval_indices[idx]].astype(int)
            return evaluator.f1_score(y_true, preds, zero_division=0)

        study_name = f"ensemble_cls_{cls.replace(' ', '_')}"
        # only delete if it exists; otherwise ignore
        try:
            optuna.delete_study(
                study_name=study_name,
                storage=optuna_db
            )
        except KeyError:
            # no existing study under that name, so nothing to delete
            pass    

        # Used by default a TPE (Tree-structured Parzen Estimator)
        study = optuna.create_study(
            study_name=f"ensemble_cls_{cls.replace(' ', '_')}",
            direction="maximize",
            storage=optuna_db,
            load_if_exists=False)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

        # best weights
        if study.best_trial:
            raw_best = [study.best_trial.params[f"w{m}_cls{idx}"] for m in range(num_models)]
            if args.no_normalize:
                best_w = np.array(raw_best, dtype=float)
            else:
                s = sum(raw_best)
                best_w = (np.array(raw_best)/s) if s > 0 else np.ones(num_models)/num_models
            best_f1 = study.best_value
        else:
            best_w = np.ones(num_models)/num_models
            best_f1 = 0.0

        weight_matrix[:, idx] = best_w
        output["F1_scores"][cls] = round(best_f1, 6)
        print(f" → weights: {best_w}   F1 = {best_f1:.4f}")


    ensemble_thresholds = {}
    for idx, cls in enumerate(eval_tasks):
        # assemble the binary votes
        bits = np.stack([
            (model_probs[m][:, eval_indices[idx]] >= thresholds_by_model[m][idx]).astype(float)
            for m in range(num_models)
        ], axis=0)  # shape (M, N)

        # weighted sum of votes
        ens_scores = np.dot(weight_matrix[:, idx], bits)

        # find best cutoff for this class
        thr_dict, _ = evaluator.find_optimal_thresholds(
            probabilities = ens_scores.reshape(-1,1),
            ground_truth  = gt_labels[:, eval_indices[idx]].reshape(-1,1),
            tasks         = [cls],
            metric        = tune_cfg.get("metric", "f1")
        )
        t_ens = thr_dict[cls]
        ensemble_thresholds[cls] = t_ens

        # optional: re‑compute F1 with this cutoff and overwrite your stored score
        final_preds = (ens_scores >= t_ens).astype(int)
        f1_final   = evaluator.f1_score(
            gt_labels[:, eval_indices[idx]].astype(int),
            final_preds,
            zero_division=0
        )
        output["F1_scores"][cls] = round(f1_final, 6)
        print(f"[POST] Class {cls}: threshold={t_ens:.3f} → F1={f1_final:.4f}")

    # finally, include these in your JSON dump:
    output["Thresholds"]["ensemble"] = {
        cls: float(round(ensemble_thresholds[cls], 6))
        for cls in eval_tasks
    }

    # final average
    avg = float(np.mean(list(output["F1_scores"].values())))
    output["F1_scores"]["Final_Average"] = round(avg, 6)

    # dump weights
    output["Weights"] = {
        model_names[m]: {
            eval_tasks[c]: float(round(weight_matrix[m, c], 6))
            for c in range(num_classes)
        }
        for m in range(num_models)
    }

    # dump thresholds
    if thresholds_by_model is not None and tune_cfg.get("stage") == "pre":
        output["Thresholds"] = {
            model_names[m]: {
                eval_tasks[c]: float(round(thresholds_by_model[m][c], 6))
                for c in range(num_classes)
            }
            for m in range(num_models)
        }
    if do_post_thr:
        output["Thresholds"] = {
            cls: float(round(ensemble_thresholds[cls], 6))
            for cls in eval_tasks
        }

    # save JSON
    with open(args.output, "w") as fo:
        json.dump(output, fo, indent=2)
    print(f"\nSaved optimized weights & thresholds to {args.output}")


if __name__ == "__main__":
    main()













# #!/usr/bin/env python3
# """
# optimize_ensemble_weights.py

# This script finds an optimal weight matrix for ensembling multiple models on the CheXpert dataset,
# maximizing the average F1-score for specified evaluation classes on the validation set.

# Usage:
#     python optimize_ensemble_weights.py --config path/to/config.json --output path/to/weights.json [--no-normalize]

# - `--config` : Path to the JSON config file containing model checkpoints and settings.
# - `--output` : Path to save the output weight matrix JSON.
# - `--no-normalize` : If set, do NOT enforce weights to sum to 1 for each class (optional; by default weights are normalized per class).

# Visualize on Optuna: optuna-dashboard sqlite:///optuna_ensemble.db
# """
# import json
# import os
# import numpy as np
# import torch
# import optuna
# import sys
# from third_party import run_models, utils
# from ensemble import evaluator
# from ensemble import model_classes
# from optuna.storages import JournalStorage
# from optuna.storages import JournalFileStorage
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Optimize ensemble weight matrix for model ensemble (maximize F1-score).")
#     parser.add_argument("--config", type=str, default="/home/fkirchhofer/repo/xai_thesis/config_optuna.json", help="Path to JSON config file with model and ensemble settings.")
#     parser.add_argument("--output", type=str, default= "/home/fkirchhofer/repo/xai_thesis/optimized_weights.json" , help="Path to save the optimized weight matrix (JSON format).")
#     parser.add_argument("--no-normalize", action="store_true", help="Disable weight normalization (by default, weights per class sum to 1).")
#     parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials per class for Optuna (default: 50).")
#     args = parser.parse_args()

#     # Load configuration
#     with open(args.config, 'r') as f:
#         config = json.load(f)
#     ensemble_cfg = config.get('ensemble', {})
#     eval_cfg = config.get('evaluation', {})

#     # Define task names (CheXpert 14-class order) and target evaluation classes
#     tasks = [
#         'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#         'Lung Opacity', 'Lung Lesion', 'Edema',
#         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
#         'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
#     ]
#     # Use evaluation_sub_tasks from config if provided, otherwise default to five CheXpert competition classes
#     eval_tasks = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
#     num_classes = len(eval_tasks)
#     num_models = len(config['models'])
#     model_names = [m['name'] for m in config['models']]

#     # Prepare models (load weights, set to eval mode) using the same procedure as run_experiments
#     # Get default arguments for model (from run_models module)
#     saved_argv = sys.argv
#     sys.argv = sys.argv[:1]  # preserve only program name to avoid interference
#     base_args = run_models.parse_arguments()  # default args for model training script
#     sys.argv = saved_argv  # restore original argv

#     models = []
#     for model_cfg in config['models']:
#         args_model = utils.copy.deepcopy(base_args) if hasattr(utils, 'copy') else __import__('copy').deepcopy(base_args)
#         # Override architecture and checkpoint as per config
#         args_model.model = model_cfg.get('architecture', args_model.model)
#         args_model.ckpt = model_cfg['checkpoint_path']

#         for key, val in model_cfg.get('overrides', {}).items():
#             setattr(args_model, key, val)
#         # Instantiate model wrapper (this loads the model checkpoint and sets .model to eval)
#         ModelWrapper = model_classes.get_model_wrapper(args_model.model)
#         model_obj = ModelWrapper(tasks=tasks, model_args=args_model)
#         models.append(model_obj)

#     # Determine whether to evaluate on test set or validation set
#     use_test = eval_cfg.get('evaluate_test_set', False)
#     # Obtain predictions from each model on the chosen dataset
#     model_probs = []  # list of numpy arrays (N_samples x N_classes) for each model



#     tune_cfg = ensemble_cfg.get('threshold_tuning')
#     thresholds = None
#     pred_ensemble_labels = None
#     thresholds_path = ensemble_cfg.get('thresholds_path')


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     for i, model in enumerate(models):
#         # Prepare the data loader (assign=True to attach to model if needed)
#         data_loader  = model.prepare_data_loader(default_data_conditions=False,
#                                                    batch_size_override=getattr(model.model_args, 'batch_size', 16),
#                                                    test_set=use_test,
#                                                    assign=True)   
#         gt_labels = None
#         if gt_labels is None:
#             all_labels = []
#             for _, batch_labels in data_loader:
#                 all_labels.append(batch_labels.numpy())
#             gt_labels = np.vstack(all_labels)   

#         # Run the model on the entire dataset to get predictions
#         logits = model.run_class_model()  # this should run through data_loader internally
#         # Convert logits to probabilities if needed
#         if eval_cfg.get('use_logits', False):
#             probs = logits.detach().cpu()
#             print(f"Model {model_names[i]}: using raw logits for evaluation.")
#         else:
#             raw_probs = torch.sigmoid(logits).detach().cpu()
#             # Per patient get view with max prob
#             probs, gt_labels = utils.get_max_prob_per_view(probs=raw_probs,
#                                                                 gt_labels=gt_labels,
#                                                                 tasks=tasks,
#                                                                 args=args_model)
            
#             # Load thresholds from npy file
#             if thresholds_path and os.path.exists(thresholds_path):
#                 thresholds = np.load(thresholds_path, allow_pickle=True).item()
#                 print(f"Loaded thresholds from {thresholds_path}")
#                 print(f"Loaded thresholds: {thresholds}")
            
#             # Compute thresholds based on F1
#             elif tune_cfg and tune_cfg.get('stage', 'post') == 'post':
#                 print(f"Enter to threshold tuning based on {tune_cfg['metric']}")       
#                 thresholds, metric_scores = evaluator.find_optimal_thresholds(
#                     probabilities=probs,
#                     ground_truth=gt_labels,
#                     tasks=tasks,
#                     metric=tune_cfg.get('metric', 'f1')
#                 )
#                 print(f"Thresholds from tuning: {thresholds}")

#             print(f"Model {model_names[i]}: using sigmoid probabilities for evaluation.")
#         model_probs.append(probs) # list(array([N,C]))        


#     if isinstance(model_probs, list):
#         model_probs = np.stack(model_probs, axis=0) # shall be -> (M, N, C)
#         print(f"Shape of model_preds after stacking: {model_probs.shape}")
#     else:
#         print("Some other format of model_probs")
    

#     # Define output weight matrix (num_models x num_eval_classes)
#     weight_matrix = np.zeros((num_models, num_classes), dtype=float)

#     # Function to compute F1 for a given set of weights on a specific class
#     def evaluate_f1_for_class(class_index, weights):
#         # Combine model probabilities for this class using the given weights.
#         # model_probs: shape (M, N, C). We take [:, :, class_index] which is (M, N).
#         class_probs = model_probs[:, :, class_index]  # (num_models, N_samples)
#         combined = np.dot(weights, class_probs)  # weighted sum over models -> (N_samples,)
#         # If not normalizing weights, combined may exceed 1; clip to [0,1] for safety
#         combined = np.clip(combined, 0.0, 1.0)
#         # Apply default threshold 0.5 to get binary predictions
#         y_true = gt_labels[:, class_index].astype(int)

#         # Compute threshold based labels
#         if thresholds is not None:

#             if isinstance(combined, np.ndarray):
#                 probs_tensor = torch.from_numpy(combined).float()
#             else:
#                 probs_tensor = combined
#             binary_preds = evaluator.threshold_based_predictions(
#                 probs=probs_tensor, 
#                 thresholds=thresholds, 
#                 tasks=tasks).cpu().numpy()

#         else:
#             print("No threshold tuning applied. Will take default threshold 0.5")
#             binary_preds = (combined >= 0.5).astype(float)

#         # Compute F1 score for this class (zero_division=0 to handle 0/0 cases)
#         f1 = evaluator.f1_score(y_true, binary_preds, zero_division=0)
#         return f1

#     # Output dict to store results
#     output_dict = {"F1_scores": {}}
#     f1_scores = []
#     optuna_db_path = "sqlite:///optuna_ensemble.db"

#     # Optuna optimization for each class
#     for idx, class_name in enumerate(eval_tasks):
#         print(f"\nOptimizing weights for class: {class_name}")
#         # Define the objective for this class
#         def objective(trial):
#             raw_w = [trial.suggest_float(f"w{m}_cls{idx}", 0.0, 1.0) for m in range(num_models)]
#             total = sum(raw_w)
#             if total > 0:
#                 weights = np.array(raw_w, dtype=float) / total
#             else:
#                 weights = np.ones(num_models, dtype=float) / num_models


#             # if args.no_normalize:
#             #     # Unnormalized: suggest independent weights in [0,1] for each model
#             #     w = np.array([trial.suggest_float(f"w{m}_cls{idx}", 0.0, 1.0) for m in range(num_models)], dtype=float)
#             #     weights = w  # no normalization
#             # else:
#             #     # Normalized: sequentially sample weights that sum to 1
#             #     w = []
#             #     remaining = 1.0
#             #     for m in range(num_models - 1):
#             #         # suggest a weight for model m between 0 and remaining
#             #         wm = trial.suggest_float(f"w{m}_cls{idx}", 0.0, remaining)
#             #         w.append(wm)
#             #         remaining -= wm
#             #     # last weight is whatever remains (could be 0)
#             #     w.append(remaining)
#             #     weights = np.array(w, dtype=float)


#             # Evaluate F1 for this class with the proposed weights
#             f1_score = evaluate_f1_for_class(tasks.index(class_name), weights)
#             return f1_score

#         # Run Optuna study for this class
#         direction = "maximize"

#         study = optuna.create_study(
#             study_name=f"ensemble_cls_{class_name.replace(' ', '_')}",
#             direction="maximize",
#             storage=optuna_db_path,
#             load_if_exists=True)

#         study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
#         best_weights = None
#         if study.best_trial:
#             # Retrieve best weight set; if normalized was enforced, ensure it sums to 1
#             if args.no_normalize:
#                 # Simply get the suggested weights from best trial
#                 best_weights = np.array([study.best_trial.params[f"w{m}_cls{idx}"] for m in range(num_models)], dtype=float)
#             else:
#                 # Reconstruct weights including the derived last weight
#                 best_params = study.best_trial.params
#                 w_temp = []
#                 # Recompute weights in order (Optuna best params holds all except the last weight implicitly)
#                 remaining = 1.0
#                 for m in range(num_models - 1):
#                     wm = best_params.get(f"w{m}_cls{idx}", 0.0)
#                     w_temp.append(wm)
#                     remaining -= wm
#                 w_temp.append(max(remaining, 0.0))
#                 best_weights = np.array(w_temp, dtype=float)
#             # Normalize weights if needed (should already be normalized unless no_normalize flag)
#             if not args.no_normalize:
#                 # Small numeric adjustments to ensure sum exactly 1
#                 total = best_weights.sum()
#                 if total > 0:
#                     best_weights = best_weights / total
#         else:
#             # Fallback: if no trial succeeded (unlikely), use equal weights
#             best_weights = np.ones(num_models, dtype=float) / num_models
#         # Store the best weights for this class in the weight matrix
#         weight_matrix[:, idx] = best_weights
#         class_f1 = study.best_value
#         f1_scores.append(class_f1)
#         output_dict['F1_scores'][class_name] = round(class_f1, 6)
#         print(f"Optimal weights for {class_name}: {best_weights} (F1 = {study.best_value:.4f})")

#     average_f1 = float(np.mean(f1_scores))
#     output_dict["F1_scores"]["Final_Average"] = round(average_f1, 6)

#     # Prepare output dictionary in desired format

#     output_dict["Weights"] = {}
#     for m_idx, model_name in enumerate(model_names):
#         output_dict["Weights"][model_name] = {}
#         for c_idx, class_name in enumerate(eval_tasks):
#             weight_val = float(round(weight_matrix[m_idx, c_idx], 6))
#             output_dict["Weights"][model_name][class_name] = weight_val

#     # Save the weight matrix as JSON
#     output_path = args.output
#     with open(output_path, 'w') as f:
#         json.dump(output_dict, f, indent=2)
#     print(f"\nSaved optimized weight matrix to {output_path}")

# if __name__ == "__main__":
#     main()
