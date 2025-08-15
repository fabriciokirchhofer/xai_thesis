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
        "--output", type=str, default= "/home/fkirchhofer/repo/xai_thesis/optimized_weights_with_multivariate_sampler.json",
        help="Path to save the optimized weights+thresholds JSON."
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="If set, do not normalize weights per class (otherwise weights sum to 1)."
    )
    parser.add_argument(
        "--trials", type=int, default=30,
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
    # get default args without interference
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

        # run model from model_classes.py
        logits = model.run_class_model().to(device)
        if eval_cfg.get("use_logits", False):
            probs = logits.detach().cpu().numpy()
            print(f"Model {model_names[i]}: using raw logits")
        else:
            raw_probs = torch.sigmoid(logits).detach().cpu()
            # reduce multiple views per patient
            probs, gt_labels = utils.get_max_prob_per_view(
                probs=raw_probs,
                gt_labels=all_gt_labels,
                tasks=tasks,
                args=model.model_args
            )
        model_probs.append(probs)
        print(f"Collected probs for model {model_names[i]} (shape {probs.shape})")

    # stack array to shape (M, N, 14)
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
    #print(f"do_post_thr: {do_post_thr}\nensemble_thresholds: {ensemble_thresholds}")


    # ----------------------------------------------------------
    # --------------- optimize weights per class ---------------
    # ----------------------------------------------------------
    weight_matrix = np.zeros((num_models, num_classes), dtype=float)
    output = {"F1_scores": {}}
    output["Thresholds"] = {}

    optuna_db = ensemble_cfg.get("optuna_db_path", "sqlite:///optuna_ensemble.db")
    for idx, cls in enumerate(eval_tasks):
        print(f"\n>>> Optimizing weights for class {cls}")
        def objective(trial):
            # Initialize trial where optuna suggests per model weights in range 0,1
            raw_w = [
                trial.suggest_float(f"w{m}_cls{idx}", 0.0, 1.0) # (name, low, high, *[, step, log])
                for m in range(num_models)
            ]
            if args.no_normalize:
                w = np.array(raw_w, dtype=float)
            else:
                # Normalize so weights across task so they sum up to 1. If raw_w too small it will do average weighting.
                s = sum(raw_w)
                w = (np.array(raw_w)/s) if s > 0 else np.ones(num_models)/num_models

            # build binary preds
            if thresholds_by_model is not None:
                # threshold each model's probs pre‑ensemble per class
                bits = np.stack([
                    (model_probs[m][:, eval_indices[idx]] >= thresholds_by_model[m][idx]).astype(float)
                    for m in range(num_models)
                ], axis=0)  # shape (M, N)
                agg = np.dot(w, bits)  # shape (N,)
                # TODO: add here again a treshold tuning for the ensemble
                #ens_thr = evaluator.find_optimal_thresholds(probabilities=agg, ground_truth=...)
                thr_dict, _ = evaluator.find_optimal_thresholds(
                                        probabilities = agg.reshape(-1, 1),
                                        ground_truth  = gt_labels[:, eval_indices[idx]].reshape(-1,1),
                                        tasks         = [cls],
                                        metric        = tune_cfg.get("metric", "f1"))
                t = thr_dict[cls]
                preds = (agg >= t).astype(int)
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
            print("Deleted study.")
        except KeyError:
            print("no existing study under that name, so nothing to delete")# no existing study under that name, so nothing to delete
            pass    

        # Used by default a TPE (Tree-structured Parzen Estimator)
        # REEEEEEEEEAAAAAADDDDD HOW THIS IS HANDELED 
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
        study = optuna.create_study(
            study_name=f"ensemble_cls_{cls.replace(' ', '_')}",
            direction="maximize",
            storage=optuna_db,
            sampler=sampler,
            load_if_exists=False)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)


        # Access for the best trial the best parameters. Otherwise default to equal weights and F1=0.
        if study.best_trial: # Optuna attribute
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


    # Do final computation with the best weight from the optimization. Follows the ensemble structure of dist_voting using soft_votes
    ensemble_thresholds = {}
    for idx, cls in enumerate(eval_tasks):
        pm = model_probs[:, :, eval_indices[idx]]  # (M, N)
        if thresholds_by_model is None:
            ens_scores = np.dot(weight_matrix[:, idx], pm)  # prob averaging
        else:
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


    # **** Dump results into JSON file
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

    # IF thr_tuning = pre. Store thresholds by model (25 values) 
    if thresholds_by_model is not None and tune_cfg.get("stage") == "pre":
        output["Thresholds"] = {
            model_names[m]: {
                eval_tasks[c]: float(round(thresholds_by_model[m][c], 6))
                for c in range(num_classes)
            }
            for m in range(num_models)
        }
    
    # IF thr_tuning = post. Store ensemble thresholds (5 values)
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

