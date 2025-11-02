import json
import os
import sys
import datetime
import copy
import shutil
import numpy as np
import pandas as pd
import torch
from third_party import run_models, utils
from ensemble import ensemble as ens_module
from ensemble import evaluator
from ensemble import model_classes  # factory for BaseModelXAI wrappers

def main():
    """
    Run an end-to-end ensemble evaluation (optionally with (a,b) grid search).

    Workflow
    --------
    1) Parse config & create a timestamped results directory (copy config for repro).
    2) Instantiate model wrappers; get per-model logits/probabilities on val/test.
    3) Build the requested ensemble strategy (average, average_voting,
       distinctiveness_weighted, distinctiveness_voting), injecting task names,
       model names, distinctiveness weights or optuna weights if present.
    4) (Optional) Grid search over (a,b) hyperparameters controlling weight sharpening
       and global scaling; aggregate per-study probabilities (max-per-view).
    5) Threshold logic:
       - 'pre'  : find per-model thresholds (for voting) and final ensemble thresholds.
       - 'none' : load thresholds from files (if provided) or use 0.5 defaults.
    6) Compute final binary predictions and metrics (AUROC, F1, Youden, Accuracy).
    7) Save artifacts (f1 grid, thresholds, labels, weight matrix if exposed, plots).

    Inputs
    ------
    config.json:
        - models[*]: architecture, checkpoint, overrides (e.g., batch_size)
        - ensemble : strategy, normalize_distinctiveness_by, threshold_tuning
        - evaluation: evaluate_test_set, use_logits (optional)
        - grid_search: perform_grid_search, grid_val_a/b, step_size

    Notes
    -----
    - Shapes:
        per-model outputs : (N_samples, C_tasks)
        stacked preds     : list[Tensor] or np.ndarray, converted inside strategies
    - Patient-level aggregation uses utils.get_max_prob_per_view.
    - If grid search is enabled, reported metrics are recomputed at best (a,b).

    Side Effects
    ------------
    - Writes artifacts under results/<experiment_name>_<timestamp>_.../
    """

    print("******************** Get run_experiments.py started ********************")
    local_time_zone = datetime.timezone(datetime.timedelta(hours=2), name="CEST")
    start = datetime.datetime.now(local_time_zone)

    parent_parser = run_models.create_parser()
    parent_parser.add_argument('--config', type=str,
        default='/home/fkirchhofer/repo/xai_thesis/config.json', # TODO: For some reason I can not pass it in the terminal. It seems unrecognized. Why?
        help='Path to JSON config file')    
    args = parent_parser.parse_args()

    # Load experiment configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 1) Build a flat context (e.g., {'method': 'LRP'})
    _ctx = utils._flatten_context(config)

    # 2) Expand placeholders everywhere they appear
    config = utils._expand_placeholders(config, _ctx)

    # 3) Normalize saliency paths: expand ~ and join base_dir with folder names
    config = utils._expand_and_join_base_dirs(config)

    # Set up timestamped results directory and copy the config there for reproducibility
    output_cfg = config.get('output', {})
    experiment_name = output_cfg.get('experiment_name', 'ensemble_experiment')
    base_results_dir = output_cfg.get('results_dir', 'results')
    timestamp = datetime.datetime.now(local_time_zone).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_results_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(results_dir, 'config.json'))

    tasks = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
             'Lung Opacity', 'Lung Lesion', 'Edema',
             'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
             'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    eval_tasks = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    # Retrieve default arguments from run_models.parse_arguments()
    # We temporarily clear sys.argv so parse_arguments() returns its defaults
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    base_args = run_models.parse_arguments()
    sys.argv = saved_argv

    # Instantiate each model configuration
    models = []
    for model_cfg in config['models']:
        args_model = copy.deepcopy(base_args)

        # Override architecture and checkpoint path per config
        args_model.model = model_cfg.get('architecture', args_model.model)
        args_model.ckpt = model_cfg['checkpoint_path']

        # Apply any additional overrides (e.g., input_size, num_classes).
        for key, val in model_cfg.get('overrides', {}).items():
               #"Make sure prepare_data_loader doesn't accept only default values.\n")
            setattr(args_model, key, val)

        # Retrieve the appropriate model wrapper class using model_classes
        ModelWrapper = model_classes.get_model_wrapper(args_model.model)
        # Instantiating the model loads weights and sets eval mode
        model_obj = ModelWrapper(tasks=tasks, model_args=args_model)
        models.append(model_obj)

    # Safety override: ensure --run_test matches config["evaluation"]["evaluate_test_set"]
    eval_cfg = config.get("evaluation", {})
    if "evaluate_test_set" in eval_cfg:
        args_model.run_test = utils._coerce_bool(eval_cfg["evaluate_test_set"])
        print(f"[INFO] Overriding args_model.run_test -> {args_model.run_test} "
            f"(from config['evaluation']['evaluate_test_set'])")

    model_probs_list = []
    use_test = config["evaluation"].get("evaluate_test_set", False) # alternative without default value config["evaluation"]["evaluate_test_set"]
    print(f"Test case: {use_test}")
    for i, model in enumerate(models):
        # Be aware to check the conditions of calling arguments here
        data_loader = models[i].prepare_data_loader(default_data_conditions=False,
                                            batch_size_override=args_model.batch_size,
                                            test_set=use_test,
                                            assign=True)
        logits = model.run_class_model()
        if config['evaluation'].get('use_logits', False):
            raw_preds = logits.cpu()
            print(f"Ensemble evaluation based on logits.")
        else:
            raw_preds = torch.sigmoid(logits).cpu()
            print(f"Ensemble evaluation based on probabilities.")
        model_probs_list.append(raw_preds)

    gt_labels = []
    for _, labels in data_loader:
        gt_labels.append(labels.numpy())
    gt_labels = np.vstack(gt_labels)

    ensemble_cfg = config.get('ensemble', {})
    ensemble_cfg['model_names'] = [m.get('name', f'Model{i+1}') for i, m in enumerate(config.get('models', []))]
    ensemble_cfg['tasks'] = tasks
    strategy_name = ensemble_cfg.get('strategy', 'average') # Get strategy by default it will take average
    strategy_fn = ens_module.StrategyFactory.get_strategy(strategy_name, **ensemble_cfg)
    per_model_voting_thresholds = None # List of single model thresholds before ensemble
    grid_cfg = config.get('grid_search', {})

    if str(grid_cfg.get('perform_grid_search')).lower() == 'true':
        do_grid_search = True
    else:
        do_grid_search = False

    if do_grid_search:
        print("Entered to grid search")
        # Extract grid parameters
        a_max = grid_cfg.get('grid_val_a', 1)
        b_max = grid_cfg.get('grid_val_b', 1)
        step = grid_cfg.get('step_size', 1)

        # Create range arrays
        grid_vals_a = np.arange(step, a_max + 1e-8, step)
        grid_vals_b = np.arange(step, b_max + 1e-8, step)
        print(f"grid_vals_a: {grid_vals_a}")
        print(f"grid_vals_b: {grid_vals_b}")
        f1_grid = np.zeros((len(grid_vals_a), len(grid_vals_b)), dtype=float)
    else:
        grid_vals_a = np.array([1.0])
        grid_vals_b = np.array([1.0])
        f1_grid = np.zeros((len(grid_vals_a), len(grid_vals_b)), dtype=float)


    # *******************************************************************
    # ******************** Ensemble structure starts ********************
    # *******************************************************************
    print(f"Go into {strategy_name} strategy.")
    tune_cfg = ensemble_cfg.get('threshold_tuning')
    ens_thresholds_path = ensemble_cfg.get('thresholds_path')
    ens_thresholds = None
    pred_ensemble_labels = None
    if do_grid_search:
        thresholds_grid = [[None for _ in range(len(grid_vals_b))] for _ in range(len(grid_vals_a))]
        pred_ensemble_labels_grid = [[None for _ in range(len(grid_vals_b))] for _ in range(len(grid_vals_a))]

    if strategy_name not in ('distinctiveness_voting', 'average_voting'):
        for i, a in enumerate(grid_vals_a):
            for j, b in enumerate(grid_vals_b):
                print(f"a:{a}\tb:{b}")

                ensemble_probs = strategy_fn(model_probs_list, a_val=a, b_val=b)

                # Per patient get view with max prob
                ensemble_probs, subset_gt_labels = utils.get_max_prob_per_view(probs=ensemble_probs,
                                                                    gt_labels=gt_labels,
                                                                    tasks=tasks,
                                                                    args=args_model)
                

                # Optuna patch - START
                # Load thresholds for TEST runs (prefer JSON, fallback to npy); else compute for VAL runs
                cfg_weights_path = ensemble_cfg.get('config_weights', None)
                if tune_cfg and tune_cfg.get('stage') == 'none' and cfg_weights_path and os.path.exists(cfg_weights_path):
                    # Read ensemble thresholds from Optuna JSON (distinctiveness_weighted: only "ensemble" is present)
                    with open(cfg_weights_path, 'r') as jf:
                        cfgw = json.load(jf)
                    thr_map = cfgw.get("Thresholds", {})
                    ens_thr_map = thr_map.get("ensemble", {})
                    ens_thresholds = {cls: float(ens_thr_map.get(cls, 0.5)) for cls in tasks}
                # Optuna patch -  END 


                elif ens_thresholds_path and os.path.exists(ens_thresholds_path):
                    # Legacy npy thresholds (still supported)
                    ens_thresholds = np.load(ens_thresholds_path, allow_pickle=True).item()
                    print(f"Loaded thresholds from {ens_thresholds_path}")

                elif tune_cfg and tune_cfg.get('stage', 'pre') == 'pre':
                    # Compute thresholds in validation mode
                    ens_thresholds, _ = evaluator.find_optimal_thresholds(
                        probabilities=ensemble_probs,
                        ground_truth=subset_gt_labels,
                        tasks=tasks,
                        metric=tune_cfg.get('metric', 'f1'))

                # Compute predictions 
                if ens_thresholds is not None:
                    #print("Use received thresholds")
                    pred_ensemble_labels = evaluator.threshold_based_predictions(
                        probs=torch.tensor(ensemble_probs),
                        thresholds=ens_thresholds,
                        tasks=tasks).numpy()
                else:
                    print("No threshold tuning applied. Will take default threshold 0.5")
                    pred_ensemble_labels = (ensemble_probs >= 0.5).astype(float)

                if do_grid_search:
                    # Evaluate F1 subset mean
                    metrics = evaluator.evaluate_metrics(
                        predictions=ensemble_probs,
                        binary_preds=pred_ensemble_labels,
                        targets=subset_gt_labels,
                        tasks=tasks,
                        metrics=['F1'],
                        evaluation_sub_tasks=eval_tasks)
                    # print(f"Returned metrics: {metrics}")
                    f1_grid[i, j] = metrics['F1_subset_mean']
                    thresholds_grid[i][j] = ens_thresholds


    # ****************** Voting structure start ******************
    else:
        # preds: list of tensors or arrays, or stacked tensor
        if isinstance(model_probs_list, list):
            if torch.is_tensor(raw_preds[0]): 
                stack = torch.stack(model_probs_list, dim=0).cpu().numpy()
            else:
                stack = np.stack(model_probs_list, axis=0)
        elif torch.is_tensor(raw_preds):
            stack = model_probs_list.cpu().numpy()
        else:
            stack = np.array(model_probs_list)

        per_model_thresholds = None
        if do_grid_search:
            per_model_thresholds_grid = [[None for _ in range(len(grid_vals_b))] for _ in range(len(grid_vals_a))]

        # Optuna patch - START
        if tune_cfg.get("stage") == 'none':
            print("Went to Test mode. Load precomputed thresholds.")
            test = True
            
            cfg_weights_path = ensemble_cfg.get('config_weights', None)
            if cfg_weights_path and cfg_weights_path.lower() != "tbd" and os.path.exists(cfg_weights_path):
                # Load per-model AND ensemble thresholds from JSON exported by optimize_ensemble_weights.py
                with open(cfg_weights_path, 'r') as jf:
                    cfgw = json.load(jf)
                thr_map = cfgw.get("Thresholds", {})
                print(f"Optuna threshold dict: {thr_map}")
                # Preserve model order as in StrategyFactory
                model_names_cfg = ensemble_cfg.get('model_names', [m["name"] for m in config["models"]])

                # per-model thresholds (shape: M x C, aligned to full `tasks`)
                per_model_thresholds_arrays = []
                for name in model_names_cfg:
                    md = thr_map.get(name, {})
                    arr = np.array([float(md.get(cls, 0.5)) for cls in tasks], dtype=float)
                    per_model_thresholds_arrays.append(arr)
                if len(per_model_thresholds_arrays) > 0:
                    per_model_thresholds = np.stack(per_model_thresholds_arrays, axis=0)  # (M, C)

                # ensemble thresholds (dict for later thresholding) 
                ens_thr_map = thr_map.get("ensemble", {})
                if isinstance(ens_thr_map, dict) and len(ens_thr_map) > 0:
                    ens_thresholds = {cls: float(ens_thr_map.get(cls, 0.5)) for cls in tasks}
            
            else:
                # If test case - thresholds from validation set shall be loaded
                per_model_thresholds_path = ensemble_cfg.get("per_model_voting_thresholds_path", None)
                if per_model_thresholds_path and per_model_thresholds_path.lower() != "tbd" and os.path.exists(ensemble_cfg.get('per_model_voting_thresholds_path')):
                    per_model_thresholds = np.load(ensemble_cfg.get('per_model_voting_thresholds_path'), allow_pickle=True)
                    print(f"\nPer model thresholds from validation run:\n{per_model_thresholds}")

        # Optuna patch - END
        else: 
            #print("Based on config params to Val mode for labels retrival and threshold creation")
            test = False

        probs_list = []
        per_model_thresholds_arrays = []
        # Loop over all models to get for each its maximum probability per study view
        for idx, model_probs in enumerate(stack):
            model_wise_probability_subset, subset_gt_labels = utils.get_max_prob_per_view(model_probs, gt_labels, tasks, args=args_model)
            probs_list.append(model_wise_probability_subset)

            if not test:
                per_model_thresholds = evaluator.find_optimal_thresholds(probabilities=probs_list[-1], 
                                                            ground_truth=subset_gt_labels,
                                                            tasks=tasks,
                                                            metric=tune_cfg.get('metric', 'f1'))[0]
                arr = np.array([per_model_thresholds[cls] for cls in tasks], dtype=float)
                per_model_thresholds_arrays.append(arr)
            else:
                model_thresholds = per_model_thresholds[idx]
                per_model_thresholds_arrays.append(model_thresholds)    

            # Compute predictions
            if per_model_thresholds is not None:
                probs_list[-1]  = evaluator.threshold_based_predictions(probs=torch.tensor(probs_list[-1]),
                                                                                thresholds=per_model_thresholds_arrays[-1],
                                                                                tasks=tasks).numpy()
            else:
                print("No threshold tuning applied. Will take default threshold 0.5 for each model")
                probs_list[-1]  = (probs_list[-1] >= 0.5).astype(float)

        votes_arr = np.stack(probs_list, axis=0)
        per_model_voting_thresholds = np.stack(per_model_thresholds_arrays, axis=0)
        # Grid search patch end************************

        # Try to load thresholds for the voting fraction from npy file
        if ens_thresholds is None and ens_thresholds_path.lower() != "tbd" and os.path.exists(ens_thresholds_path):
            ens_thresholds = np.load(ens_thresholds_path, allow_pickle=True).item()
            print(f"\nLoaded precomputed ensemble thresholds for the voting fraction (npy).\n {ens_thresholds}")

        for i, a in enumerate(grid_vals_a):
            for j, b in enumerate(grid_vals_b):
                print(f"a:{a}\tb:{b}")
                weighted_vote_fraction = strategy_fn(votes_arr, all_targets=subset_gt_labels, a_val=a, b_val=b)
                ensemble_probs = weighted_vote_fraction

                # Compute thresholds based on F1
                if tune_cfg and tune_cfg.get('stage', 'pre') == 'pre':
                    ens_thresholds, _ = evaluator.find_optimal_thresholds(
                        probabilities=weighted_vote_fraction,
                        ground_truth=subset_gt_labels,
                        tasks=tasks,
                        metric=tune_cfg.get('metric', 'f1'))
                # Compute threshold based labels
                if ens_thresholds is not None:
                    pred_ensemble_labels = evaluator.threshold_based_predictions(
                        probs=torch.tensor(weighted_vote_fraction),
                        thresholds=ens_thresholds,
                        tasks=tasks).numpy()
                else:
                    print("No threshold tuning applied. Will take default threshold 0.5")
                    pred_ensemble_labels = (ensemble_probs >= 0.5).astype(float)

                if do_grid_search:
                    # Evaluate F1 subset mean
                    metrics = evaluator.evaluate_metrics(
                        predictions=ensemble_probs,
                        binary_preds=pred_ensemble_labels,
                        targets=subset_gt_labels,
                        tasks=tasks,
                        metrics=['F1'],
                        evaluation_sub_tasks=eval_tasks)
                    #print(f"Returned metrics: {metrics}")
                    f1_grid[i, j] = metrics['F1_subset_mean']
                    thresholds_grid[i][j] = ens_thresholds
                    per_model_thresholds_grid[i][j] = per_model_voting_thresholds

    # ****************** Voting structure ends ******************

    results = evaluator.evaluate_metrics(
        predictions=ensemble_probs,
        binary_preds=pred_ensemble_labels,
        targets=subset_gt_labels,
        use_logits=eval_cfg.get('use_logits', False),
        metrics=eval_cfg.get('metrics', ['AUROC']),
        evaluation_sub_tasks=eval_cfg.get('evaluation_sub_tasks', eval_tasks),
        tasks=tasks) 
    print(f"CHECK - Evaluation completed with ensemble probs shape: {ensemble_probs.shape} in mode: {args_model.run_test}")

    if do_grid_search:
        print(f"Final f1_grid: {f1_grid}")
        print("Retrieve the best a and b vals")
        # Find best (a, b)
        best_idx = np.unravel_index(np.nanargmax(f1_grid), f1_grid.shape)
        ens_thresholds = thresholds_grid[best_idx[0]][best_idx[1]]
        if strategy_name in ('distinctiveness_voting', 'average_voting'):
            per_model_voting_thresholds = per_model_thresholds_grid[best_idx[0]][best_idx[1]]
        best_a = float(grid_vals_a[best_idx[0]])
        best_b = float(grid_vals_b[best_idx[1]])
        best_f1 = float(f1_grid[best_idx]) 
        print(f"Best a: {best_a}")
        print(f"Best b: {best_b}")
        print(f"Best F1 score: {best_f1}")
        results["grid_search"] = {
            "best_a": best_a,
            "best_b": best_b,
            "best_f1_subset_mean": best_f1}

    # Plot graph which shows the probabilities and where the threshold for the binary prediction is
    utils.plot_threshold_effects(ensemble_probs=ensemble_probs,
                            binary_preds=pred_ensemble_labels,
                            thresholds=ens_thresholds,
                            class_names=tasks,
                            save_path=os.path.join(results_dir, "plots/threshold_effects"))

    evaluator.plot_roc(predictions=ensemble_probs,
                    ground_truth=subset_gt_labels,
                    tasks=tasks,
                    save_dir=results_dir)

    model_names = [m["_identifier"] for m in config["models"]]

    if output_cfg.get('plot_models_analysis', False):
        print("Plot ensemble models analysis")
        evaluator.plot_prediction_distributions(model_probs=model_probs_list,
                                                tasks=tasks,
                                                class_idx=None,
                                                sample_idx=None,
                                                bins=20,
                                                kde=True,
                                                model_names=model_names,
                                                save_dir=results_dir)

        evaluator.plot_prediction_distributions(model_probs=model_probs_list,
                                                tasks=tasks,
                                                class_idx=None,
                                                sample_idx=None,
                                                bins=20,
                                                kde=False,
                                                model_names=model_names,
                                                save_dir=results_dir)

        evaluator.plot_model_correlation(model_probs=model_probs_list,
                                        model_names=model_names,
                                        save_dir=results_dir)

        # evaluator.plot_umap_model_predictions(model_probs=model_probs,
        #                                 model_names=model_names,
        #                                 n_neighbors=30,
        #                                 min_dist=0.0,
        #                                 metric=eval_cfg.get('umap_metric', 'euclidean'),
        #                                 n_components=3,
        #                                 save_dir=results_dir) # scp -r fkirchhofer@nerve.artorg.unibe.ch:nerve_folder_path /Users/fabri/Desktop

        distinctiveness_files = ensemble_cfg.get('distinctiveness_files', None)
        if ensemble_cfg.get('config_weights', "tbd") == "tbd" and os.path.exists(distinctiveness_files[0]):
            utils.plot_distinctiveness_heatmap_from_files(distinctiveness_files=distinctiveness_files,
                                                        models=model_names,
                                                        cmap="plasma",
                                                        save_path=results_dir)

            utils.plot_distinctiveness_radar_from_files(distinctiveness_files=distinctiveness_files,
                                                        models=model_names,
                                                        save_path=results_dir)
        if do_grid_search:
            utils.plot_search_grid_heatmap(b_vals=grid_vals_b,
                                       a_vals=grid_vals_a,
                                       f1_grid=f1_grid,
                                       results_dir=results_dir)



    # if we used distinctiveness_weighted, save its weight matrix
    if hasattr(strategy_fn, 'weight_matrix'):
        wm = strategy_fn.weight_matrix  # this is a NumPy array of shape (n_models, n_classes)
        wm_path = os.path.join(results_dir, 'distinctiveness_weight_matrix.json')

        # convert to nested lists so it's JSON‐serializable
        with open(wm_path, 'w') as wm_file:
            json.dump(wm.tolist(), wm_file, indent=2)
        
        # Also save as CSV with labeled models and classes
        wm_csv_path = os.path.join(results_dir, 'distinctiveness_weight_matrix.csv')
        # Get model names from ensemble_cfg, or fallback to generic names
        model_names_labels = ensemble_cfg.get('model_names', [f'Model_{chr(65+i)}' for i in range(len(wm))])
        df = pd.DataFrame(wm, index=model_names_labels, columns=tasks)
        df.to_csv(wm_csv_path, index=True)
        print(f"Distinctiveness weight matrix saved as CSV to: {wm_csv_path}")
        
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as mf:
        json.dump(results, mf, indent=4)
    np.save(os.path.join(results_dir, 'ensemble_probs.npy'), ensemble_probs)
    np.save(os.path.join(results_dir, 'GT_labels.npy'), subset_gt_labels)
    if do_grid_search:
        np.save(os.path.join(results_dir, 'f1_grid.npy'), f1_grid)
    if pred_ensemble_labels is not None:
        np.save(os.path.join(results_dir, 'ensemble_labels.npy'), pred_ensemble_labels)
    if ens_thresholds is not None:
        np.save(os.path.join(results_dir, 'thresholds.npy'), ens_thresholds)
    if per_model_voting_thresholds is not None:
        np.save(os.path.join(results_dir, 'per_model_voting_thresholds.npy'), per_model_voting_thresholds)

    print(f"Experiment complete. Results saved in {results_dir}")

    # Time measurement
    end = datetime.datetime.now(local_time_zone)
    delta = end-start
    total_seconds = int(delta.total_seconds())
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"Elapsed time: {hours}h {minutes}m {seconds}s")


if __name__ == '__main__':
    main()
