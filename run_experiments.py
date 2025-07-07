import json
import os
import sys
import datetime
import copy
import shutil
import numpy as np
import torch
from third_party import run_models, utils
from ensemble import ensemble as ens_module
from ensemble import evaluator
from ensemble import model_classes  # factory for BaseModelXAI wrappers

def main():
    print("******************** Get run_experiments.py started ********************")
    local_time_zone = datetime.timezone(datetime.timedelta(hours=2), name="CEST")
    start = datetime.datetime.now(local_time_zone)

    parent_parser = run_models.create_parser()
    parent_parser.add_argument(
        '--config',        
        type=str,
        default='config.json', # TODO: For some reason I can not pass it in the terminal. It seems unrecognized. Why?
        help='Path to JSON config file'
    )
    args = parent_parser.parse_args()

    # Load experiment configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

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
        # Copy default args to preserve base settings for each model
        args_model = copy.deepcopy(base_args)

        # Override architecture and checkpoint path per config
        args_model.model = model_cfg.get('architecture', args_model.model)
        args_model.ckpt = model_cfg['checkpoint_path']

        # Apply any additional overrides (e.g., input_size, num_classes). 
        for key, val in model_cfg.get('overrides', {}).items():
            # set attribute by name into args_model
            #print(f"For {model_cfg['name']}:{args_model.model} override following default arguments with:\tkey:{key}\tval:{val}.\n"
                  #"Make sure prepare_data_loader doesn't accept only default values.\n")
            setattr(args_model, key, val)

        # Retrieve the appropriate model wrapper class using model_classes
        ModelWrapper = model_classes.get_model_wrapper(args_model.model)
        # Instantiating the model loads weights and sets eval mode
        model_obj = ModelWrapper(tasks=tasks, model_args=args_model)
        models.append(model_obj)

    model_preds = []
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
            preds = logits.cpu()
            print(f"Ensemble evaluation based on logits.")
        else:
            preds = torch.sigmoid(logits).cpu()
            print(f"Ensemble evaluation based on probabilities.")
        model_preds.append(preds)

    gt_labels = []
    for _, labels in data_loader:
        gt_labels.append(labels.numpy())
    gt_labels = np.vstack(gt_labels)    

    ensemble_cfg = config.get('ensemble', {})
    strategy_name = ensemble_cfg.get('strategy', 'average') # Get strategy by default it will take average
    
# ****************** Distinctiveness voting ensemble start ******************
    if strategy_name == 'distinctiveness_voting':
        tune_cfg = ensemble_cfg.get('threshold_tuning')
        pred_ensemble_labels = None
        thresholds = None     # Thresholds for ensemble per class 
        per_model_voting_thresholds = None # List of single model thresholds before ensemble
        thresholds_path = ensemble_cfg.get('thresholds_path')
    
        strategy_fn = ens_module.StrategyFactory.get_strategy(strategy_name, 
                                                              **ensemble_cfg, 
                                                              all_targets=gt_labels)
        weighted_vote_fraction, gt_labels, per_model_voting_thresholds = strategy_fn(model_preds, all_targets=gt_labels)
        ensemble_preds = weighted_vote_fraction
                
        # Load thresholds from npy file
        if thresholds_path and os.path.exists(thresholds_path):
            thresholds = np.load(thresholds_path, allow_pickle=True).item()
            print(f"Loaded following thresholds from the voting fraction ensemble: {thresholds}")
        
        # Compute thresholds based on F1
        elif tune_cfg and tune_cfg.get('stage', 'post') == 'post':
            print(f"Enter to ensemble voting threshold tuning based on {tune_cfg['metric']}")       
            thresholds, metric_scores = evaluator.find_optimal_thresholds(
                probabilities=weighted_vote_fraction,
                ground_truth=gt_labels,
                tasks=tasks,
                metric=tune_cfg.get('metric', 'f1')
            )
            print(f"Thresholds for weighted voting fraction ensemble from tuning: {thresholds}")

        # Compute threshold based labels
        if thresholds is not None:
            pred_ensemble_labels = evaluator.threshold_based_predictions(
                probs=torch.tensor(weighted_vote_fraction), 
                thresholds=thresholds, 
                tasks=tasks
            ).numpy()

        else:
            print("No threshold tuning applied. Will take default threshold 0.5")
            pred_ensemble_labels = (ensemble_preds >= 0.5).astype(float)
# ****************** Distinctiveness voting ensemble end ******************
        
    else:
        strategy_fn = ens_module.StrategyFactory.get_strategy(strategy_name, **ensemble_cfg)  
        ensemble_preds = strategy_fn(model_preds)

        # Per patient get view with max prob
        ensemble_preds, gt_labels = utils.get_max_prob_per_view(probs=ensemble_preds,
                                                            gt_labels=gt_labels,
                                                            tasks=tasks,
                                                            args=args_model)

        # ************************ Threshold handling start ************************
        tune_cfg = ensemble_cfg.get('threshold_tuning')
        thresholds = None
        pred_ensemble_labels = None
        thresholds_path = ensemble_cfg.get('thresholds_path')

        # Load thresholds from npy file
        if thresholds_path and os.path.exists(thresholds_path):
            thresholds = np.load(thresholds_path, allow_pickle=True).item()
            print(f"Loaded thresholds from {thresholds_path}")
            print(f"Loaded thresholds: {thresholds}")
        
        # Compute thresholds based on F1
        elif tune_cfg and tune_cfg.get('stage', 'post') == 'post':
            print(f"Enter to threshold tuning based on {tune_cfg['metric']}")       
            thresholds, metric_scores = evaluator.find_optimal_thresholds(
                probabilities=ensemble_preds,
                ground_truth=gt_labels,
                tasks=tasks,
                metric=tune_cfg.get('metric', 'f1')
            )
            print(f"Thresholds from tuning: {thresholds}")

        # Compute threshold based labels
        if thresholds is not None:
            pred_ensemble_labels = evaluator.threshold_based_predictions(
                probs=torch.tensor(ensemble_preds), 
                thresholds=thresholds, 
                tasks=tasks
            ).numpy()

        else:
            print("No threshold tuning applied. Will take default threshold 0.5")
            pred_ensemble_labels = (ensemble_preds >= 0.5).astype(float)
        # ************************ Threshold handling end ************************


    eval_cfg = config.get('evaluation', {})
    results = evaluator.evaluate_metrics(
        predictions=ensemble_preds,
        binary_preds=pred_ensemble_labels,
        targets=gt_labels,
        use_logits=eval_cfg.get('use_logits', False),
        metrics=eval_cfg.get('metrics', ['AUROC']),
        evaluation_sub_tasks=eval_cfg.get('evaluation_sub_tasks', eval_tasks),
        tasks=tasks
    )
    evaluator.plot_roc(predictions=ensemble_preds,
                    ground_truth=gt_labels,
                    tasks=tasks,
                    save_dir=results_dir)
    
    model_names = [m["_identifier"] for m in config["models"]]
    #ensemble_models_analysis = output_cfg.get('plot_models_analysis', False)
    
    if output_cfg.get('plot_models_analysis', False):
        print("Plot ensemble models analysis")
        evaluator.plot_prediction_distributions(model_preds=model_preds,
                                                tasks=tasks,
                                                class_idx=None,
                                                sample_idx=None,
                                                bins=20,
                                                kde=True,
                                                model_names=model_names,
                                                save_dir=results_dir)
        
        evaluator.plot_prediction_distributions(model_preds=model_preds,
                                                tasks=tasks,
                                                class_idx=None,
                                                sample_idx=None,
                                                bins=20,
                                                kde=False,
                                                model_names=model_names,
                                                save_dir=results_dir)

        evaluator.plot_model_correlation(model_preds=model_preds,
                                        model_names=model_names,
                                        save_dir=results_dir)
        
        # evaluator.plot_umap_model_predictions(model_preds=model_preds,
        #                                 model_names=model_names,
        #                                 n_neighbors=30,
        #                                 min_dist=0.0,
        #                                 metric=eval_cfg.get('umap_metric', 'euclidean'),
        #                                 n_components=3,
        #                                 save_dir=results_dir) # scp -r fkirchhofer@nerve.artorg.unibe.ch:nerve_folder_path /Users/fabri/Desktop

    # if we used distinctiveness_weighted, save its weight matrix
    if strategy_name.lower() == 'distinctiveness_weighted' and hasattr(strategy_fn, 'weight_matrix'):
        wm = strategy_fn.weight_matrix  # this is a NumPy array of shape (n_models, n_classes)
        wm_path = os.path.join(results_dir, 'distinctiveness_weight_matrix.json')
        # convert to nested lists so it’s JSON‐serializable
        with open(wm_path, 'w') as wm_file:
            json.dump(wm.tolist(), wm_file, indent=2)
        print(f"Saved distinctiveness weight_matrix to {wm_path}")  

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as mf:
        json.dump(results, mf, indent=4)
    np.save(os.path.join(results_dir, 'ensemble_preds.npy'), ensemble_preds)
    if pred_ensemble_labels is not None:
        np.save(os.path.join(results_dir, 'ensemble_labels.npy'), pred_ensemble_labels)
    np.save(os.path.join(results_dir, 'true_labels.npy'), gt_labels)
    if thresholds is not None:
        np.save(os.path.join(results_dir, 'thresholds.npy'), thresholds)
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
