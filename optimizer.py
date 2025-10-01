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
import sys
import copy
import numpy as np
import torch
import optuna

from third_party import run_models, utils
from ensemble import evaluator, model_classes

optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    """
    Optimize per-class ensemble weights (and optionally thresholds) with Optuna.

    Workflow
    --------
    1) Load config (models, ensemble strategy, threshold tuning, evaluation subset).
    2) Build model wrappers and collect per-model predictions on validation or test.
    3) For each eval class, run an Optuna study to find a weight vector (one weight per model),
       optionally normalizing weights (sum to 1) unless --no-normalize is given.
    4) Inside the objective, compute the ensemble output (weighted sum for probabilities or
       weighted vote fraction for voting), reduce to per-study probabilities (max per view),
       apply thresholds (pre: tuned per trial, otherwise default 0.5), and return the F1.
    5) After optimization, (re)compute ensemble with best weights and find final class-wise
       thresholds (when applicable). Save JSON with Weights, Thresholds, and F1 scores.

    Notes
    -----
    - Shapes:
        model_probs_arr: (M_models, N_samples, C_tasks)
        eval_indices: list[int] mapping evaluation_sub_tasks -> indices in full 14 tasks
    - Study objective maximizes F1 on the selected class.
    - Thresholds for 'stage="none"' default to 0.5 inside the objective for safety.

    Side Effects
    ------------
    - Writes an output JSON file containing "Weights", "Thresholds", and "F1_scores".

    CLI
    ---
    --config: path to config.json
    --output: output JSON path
    --no-normalize: disable per-class weight normalization
    --trials: trials per class (default 300)
    """

    import argparse
    parser = argparse.ArgumentParser(description="Optimize ensemble weight matrix (and thresholds).")
    
    parser.add_argument("--config", type=str, default="/home/fkirchhofer/repo/xai_thesis/config.json",
        help="Path to JSON config file with model and ensemble settings.")
    
    parser.add_argument("--output", type=str, default= "/home/fkirchhofer/repo/xai_thesis/optimized_weights_with_multivariate_sampler_TEMP.json",
        help="Path to save the optimized weights+thresholds JSON.")
    
    parser.add_argument("--no-normalize", action="store_true",
        help="If set, do not normalize weights per class (otherwise weights sum to 1).")
    
    parser.add_argument("--trials", type=int, default=300,
        help="Number of Optuna trials per class (default: 30).")
    
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # load config
    with open(args.config, "r") as f:
        config = json.load(f)
    ensemble_cfg = config.get("ensemble", {})
    eval_cfg = config.get("evaluation", {})
    tune_cfg = ensemble_cfg.get("threshold_tuning", {})
    ens_strategy = ensemble_cfg.get("strategy", "distinctiveness_voting")
    ensemble_cfg['normalize_distinctiveness_by'] = 'task' # Ensure normalization happens accross tasks 

    # define tasks
    # full CheXpert 14 tasks order
    tasks = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema",
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
    
    # evaluation classes (override via config if provided)
    eval_tasks = config.get("evaluation_sub_tasks",
        ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"])

    num_classes = len(eval_tasks)
    num_models = len(config["models"])
    model_names = [m["name"] for m in config["models"]]

    # prepare models 
    # get default args without interference
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    base_args = run_models.parse_arguments()
    sys.argv = saved_argv

    models = []
    for m_cfg in config["models"]:
        m_args = copy.deepcopy(base_args)
        m_args.model = m_cfg.get("architecture", m_args.model)
        m_args.ckpt = m_cfg["checkpoint_path"]
        for k, v in m_cfg.get("overrides", {}).items():
            setattr(m_args, k, v)
        Wrapper = model_classes.get_model_wrapper(m_args.model)
        models.append(Wrapper(tasks=tasks, model_args=m_args))


    # *********************************************************************
    # ****************** Get probabilities + true labels ******************
    # *********************************************************************
    use_test = eval_cfg.get("evaluate_test_set", False)
    
    model_probs = []
    all_gt_labels = None 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, model in enumerate(models):
        # prepare loader
        data_loader = model.prepare_data_loader(
            default_data_conditions=False,
            batch_size_override=getattr(model.model_args, "batch_size", 16),
            test_set=use_test,
            assign=True)
        logits = model.run_class_model()
        raw_probs = torch.sigmoid(logits).cpu()
        model_probs.append(raw_probs)      
        
        # collect GT once
        if all_gt_labels is None:
            all_lbls = []
            for _, batch_lbl in data_loader:
                all_lbls.append(batch_lbl.numpy())
            all_gt_labels = np.vstack(all_lbls)

    # indices of eval_tasks within the full tasks list
    eval_indices = [tasks.index(c) for c in eval_tasks]
    
    # stack to a single ndarray: (M, N, 14) - Used for both strategies the same way
    model_probs_arr = np.stack(model_probs, axis=0)


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

            # Dist weighted section - START
            if ens_strategy == "distinctiveness_weighted":

                pm = model_probs_arr[:, :, eval_indices[idx]] # (M, N)
                ensemble_probs = np.dot(w, pm)

                # reduce multiple views per patient
                ensemble_probs_view, subset_gt_labels = utils.get_max_prob_per_view(
                    probs=ensemble_probs.reshape(-1,1),
                    gt_labels=all_gt_labels[:, eval_indices[idx]].reshape(-1,1),
                    tasks=[cls],
                    args=model.model_args)

                if tune_cfg.get("stage", "pre") == "pre":
                    thr_dict, _ = evaluator.find_optimal_thresholds(
                            probabilities=ensemble_probs_view,
                            ground_truth=subset_gt_labels,
                            tasks=[cls],
                            metric=tune_cfg.get("metric", "f1"))
                    t = float(thr_dict[cls])
                    preds = (ensemble_probs_view >= t).astype(int)
                else:
                    preds = (ensemble_probs_view >= 0.5).astype(int)
                
                #y_true = subset_gt_labels[:, eval_indices[idx]].astype(int)
                return evaluator.f1_score(subset_gt_labels, preds, zero_division=0)
            # Dist weighted section - END

            # Voting ens structure - START
            elif ens_strategy == "distinctiveness_voting":

                pm = model_probs_arr[:, :, eval_indices[idx]]  # (M, N)
                
                votes_list = []
                gt_view = None

                for m in range(num_models):
                    # reduce multiple views per patient
                    model_max_probs, gt_model = utils.get_max_prob_per_view(
                        probs=pm[m].reshape(-1,1),
                        gt_labels=all_gt_labels[:, eval_indices[idx]].reshape(-1,1),
                        tasks=[cls],
                        args=model.model_args)

                    gt_view = gt_model if gt_view is None else gt_view # gt_model (200,1)
                    # Threshold tuning per model
                    thresholds_by_model = None
                    if tune_cfg.get("stage", "pre") == "pre":
                        thresholds_by_model_dict, _ = evaluator.find_optimal_thresholds(
                            probabilities=model_max_probs,
                            ground_truth=gt_model,
                            tasks=[cls],
                            metric=tune_cfg.get("metric", "f1")) 
                        thresholds_by_model = thresholds_by_model_dict[cls] # Get value out of dict                   

                    if thresholds_by_model is not None:
                        # threshold each model's probs pre-ensemble per class
                        model_votes = (model_max_probs >= thresholds_by_model).astype(float)                        
                    else:
                        model_votes = (model_max_probs >= 0.5).astype(float)

                    votes_list.append(model_votes.reshape(-1))

                votes_arr = np.stack(votes_list, axis=0) # (M, N)               
                soft_vote = np.dot(w, votes_arr)  # (N,)

                # Threshold tuning per model
                if tune_cfg.get("stage", "pre") == "pre":
                    # Always tune the ensemble cutoff inside the objective for consistency
                    thr_dict, _ = evaluator.find_optimal_thresholds(
                        probabilities=soft_vote.reshape(-1, 1),
                        ground_truth=gt_view,
                        tasks=[cls],
                        metric=tune_cfg.get("metric", "f1"))
                    t = float(thr_dict[cls])              
                else:
                    t = 0.5                
                preds = (soft_vote >= t).astype(int)
                #y_true = subset_gt_labels[:, eval_indices[idx]].astype(int)
                return evaluator.f1_score(gt_view.ravel().astype(int), preds, zero_division=0) # subset_gt_labels / gt_view.ravel().astype(int)
                    # Voting ens structure - END

            else:
                raise ValueError(f"Unknown ensemble strategy: {ens_strategy}")



        study_name = f"ensemble_cls_{cls.replace(' ', '_')}"
        # only delete if it exists; otherwise ignore
        try:
            optuna.delete_study(study_name=study_name, storage=optuna_db)
            print("Deleted study.")
        except Exception:
            print("no existing study under that name, so nothing to delete")

        # Used by default a TPE (Tree-structured Parzen Estimator)
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        study = optuna.create_study(
            study_name=f"ensemble_cls_{cls.replace(' ', '_')}",
            direction="maximize",
            storage=optuna_db,
            sampler=sampler,
            load_if_exists=False)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)


        # best weights for this class
        if study.best_trial is not None:
            raw_best = [study.best_trial.params[f"w{m}_cls{idx}"] for m in range(num_models)]
            if args.no_normalize:
                best_w = np.array(raw_best, dtype=float)
            else:
                s = sum(raw_best)
                best_w = (np.array(raw_best)/s) if s > 0 else np.ones(num_models)/num_models
            best_f1 = float(study.best_value)
        else:
            best_w = np.ones(num_models) / num_models
            best_f1 = 0.0

        weight_matrix[:, idx] = best_w
        output["F1_scores"][cls] = round(best_f1, 6)
        print(f" → weights: {best_w}   F1 = {best_f1:.4f}")


    # -------------------------------
    # Final ensemble thresholds (per class) using best weights
    # -------------------------------
    ensemble_thresholds = {}
    thresholds_by_model = None
    for idx, cls in enumerate(eval_tasks):
        model_probs_arr = np.stack(model_probs, axis=0)
        pm = model_probs_arr[:, :, eval_indices[idx]]  # (M, N)

        if ens_strategy == "distinctiveness_weighted":
            ens_scores = np.dot(weight_matrix[:, idx], pm)  # weighted prob averaging (N,)
            ens_view, gt_view = utils.get_max_prob_per_view(
                probs=ens_scores.reshape(-1, 1),
                gt_labels=all_gt_labels[:, eval_indices[idx]].reshape(-1, 1),
                tasks=[cls],
                args=base_args
            )


        elif ens_strategy == "distinctiveness_voting":
            if thresholds_by_model is None:
                thresholds_by_model = [[] for _ in range(num_models)]
            bits_list = []
            gt_view = None
            for m in range(num_models):
                model_max_probs, gt_model = utils.get_max_prob_per_view(
                    probs=pm[m].reshape(-1, 1),
                    gt_labels=all_gt_labels[:, eval_indices[idx]].reshape(-1, 1),
                    tasks=[cls],
                    args=base_args
                )
                gt_view = gt_model if gt_view is None else gt_view
                if tune_cfg.get("stage", "pre") == "pre":
                    thr_m_dict, _ = evaluator.find_optimal_thresholds(
                        probabilities=model_max_probs,
                        ground_truth=gt_model,
                        tasks=[cls],
                        metric=tune_cfg.get("metric", "f1")
                    )
                    t_m = float(thr_m_dict[cls])
                    thresholds_by_model[m].append(t_m)
                else:
                    t_m = 0.5
                bits_list.append((model_max_probs.ravel() >= t_m).astype(float))
            votes_arr = np.stack(bits_list, axis=0)  # (M, N)
            ens_view = np.dot(weight_matrix[:, idx], votes_arr).reshape(-1, 1)

        else:
            raise ValueError(f"Unknown ensemble strategy: {ens_strategy}")

        # find best cutoff for this class on final ensemble scores
        thr_dict, _ = evaluator.find_optimal_thresholds(
            probabilities=ens_view,
            ground_truth=gt_view,
            tasks=[cls],
            metric=tune_cfg.get("metric", "f1"))
        
        t_ens = float(thr_dict[cls])
        ensemble_thresholds[cls] = t_ens

        # compute final F1 with this cutoff (store as the official class score)
        final_preds = (ens_view.ravel() >= t_ens).astype(int)
        f1_final = evaluator.f1_score(gt_view.ravel().astype(int), final_preds, zero_division=0)



        output["F1_scores"][cls] = round(float(f1_final), 6)
        print(f"[POST] Class {cls}: threshold={t_ens:.3f} → F1={f1_final:.4f}")



    # -------------------------------
    # Serialize thresholds according to STRATEGY
    # -------------------------------
    thresholds_out = {}
    if ens_strategy == "distinctiveness_weighted":
        # Ensemble thresholds per class
        thresholds_out["ensemble"] = {cls: float(round(ensemble_thresholds[cls], 6))
            for cls in eval_tasks}
        

    elif ens_strategy == "distinctiveness_voting":
        # Per-model thresholds
        if thresholds_by_model is not None and tune_cfg.get("stage") == "pre":
            for m_idx, m_name in enumerate(model_names):
                thresholds_out[m_name] = {eval_tasks[c]: float(round(thresholds_by_model[m_idx][c], 6))
                    for c in range(num_classes)}
        
        # Ensemble thresholds
        thresholds_out["ensemble"] = {cls: float(round(ensemble_thresholds[cls], 6))
            for cls in eval_tasks}
    else:
        raise ValueError(f"Unknown ensemble strategy: {ens_strategy}")
    output["Thresholds"] = thresholds_out

    # -------------------------------
    # Weights & Average F1
    # -------------------------------
    avg = float(np.mean(list(output["F1_scores"].values())))
    output["F1_scores"]["Final_Average"] = round(avg, 6)

    output["Weights"] = {
        model_names[m]: {
            eval_tasks[c]: float(round(weight_matrix[m, c], 6))
            for c in range(num_classes)
        }
        for m in range(num_models)
    }
    # save JSON
    with open(args.output, "w") as fo:
        json.dump(output, fo, indent=2)
    print(f"\nSaved optimized weights & thresholds to {args.output}")

if __name__ == "__main__":
    main()

