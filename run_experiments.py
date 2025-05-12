import json
import os
import sys
import datetime
import copy
import shutil
import numpy as np
import torch
from third_party import run_models
from ensemble import ensemble as ens_module
from ensemble import evaluator
from ensemble import model_classes  # factory for BaseModelXAI wrappers


def main():
    print("******************** Get run_experiments started ********************")
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

    tasks = [
        'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion' , 'Edema' ,
        'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    eval_tasks = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    # Retrieve default arguments from run_models.parse_arguments()
    # We temporarily clear sys.argv so parse_arguments() returns its defaults
    saved_argv = sys.argv
    print(f"Saved argv: {saved_argv}")
    sys.argv = sys.argv[:1]
    base_args = run_models.parse_arguments()
    print(f"Base args: {base_args}")
    sys.argv = saved_argv
    print(f"Updated argv: {sys.argv}")

    # Instantiate each model configuration
    models = []
    for model_cfg in config['models']:
        print(f"Current m_cfg is: {model_cfg}")
        # Copy default args to preserve base settings for each model
        args_model = copy.deepcopy(base_args)
        print(f"*** args_model are {args_model}")

        # Override architecture and checkpoint path per config
        args_model.model = model_cfg.get('architecture', args_model.model)
        args_model.ckpt = model_cfg['checkpoint_path']

        # Apply any additional overrides (e.g., input_size, num_classes). 
        for key, val in model_cfg.get('overrides', {}).items():
            # set attribute by name into args_model
            setattr(args_model, key, val)

        # Retrieve the appropriate model wrapper class using model_classes
        print(f"Model name: {args_model.model}")
        ModelWrapper = model_classes.get_model_wrapper(args_model.model)
        # Instantiating the model loads weights and sets eval mode
        model_obj = ModelWrapper(tasks=tasks, model_args=args_model)
        models.append(model_obj)

    model_preds = []
    for i, model in enumerate(models):
        data_loader = models[i].prepare_data_loader(default_data_conditions=True,
                                            batch_size_override=None,
                                            test_set=False,
                                            assign=True)
        logits = model.run_class_model()
        if config['evaluation'].get('use_logits', False):
            preds = logits.cpu()
            print(f"Ensembly evaluation based on logits.")
        else:
            preds = torch.sigmoid(logits).cpu()
            print(f"Ensembly evaluation based on probabilities")
        model_preds.append(preds)

    ensemble_cfg = config.get('ensemble', {})
    strategy_name = ensemble_cfg.get('strategy', 'average') # Get strategy by default it will take average
    strategy_fn = ens_module.StrategyFactory.get_strategy(strategy_name, **ensemble_cfg)
    ensemble_preds = strategy_fn(model_preds)

    # Get labels for evaluation
    all_targets = []
    for _, labels in data_loader:
        all_targets.append(labels.numpy())
    all_targets = np.vstack(all_targets)

    tune_cfg = ensemble_cfg.get('threshold_tuning')
    thresholds = None
    ensemble_labels = None
    if tune_cfg and tune_cfg.get('stage', 'post') == 'post':
        print(f"Enter to threshold tuning based on {tune_cfg['metric']}")
        
        thresholds, _ = evaluator.find_optimal_thresholds(
            probabilities=ensemble_preds,
            ground_truth=all_targets,
            tasks=tasks,
            metric=tune_cfg.get('metric', 'youden')
        )
        
        ensemble_labels = evaluator.threshold_based_predictions(
            probs=torch.tensor(ensemble_preds), 
            thresholds=thresholds, 
            tasks=tasks
        ).numpy()
    else:
        ensemble_labels = (ensemble_preds >= 0.5).astype(float)

    eval_cfg = config.get('evaluation', {})
    results = evaluator.evaluate_metrics(
        predictions=ensemble_preds,
        binary_preds=ensemble_labels,
        targets=all_targets,
        use_logits=eval_cfg.get('use_logits', False),
        metrics=eval_cfg.get('metrics', ['AUROC']),
        average_auroc_classes=eval_cfg.get('average_auroc_classes', eval_tasks),
        tasks=tasks
    )
    evaluator.plot_roc(predictions=ensemble_preds,
                    ground_truth=all_targets,
                    tasks=tasks,
                    save_dir=results_dir)

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as mf:
        json.dump(results, mf, indent=4)
    np.save(os.path.join(results_dir, 'ensemble_preds.npy'), ensemble_preds)
    if ensemble_labels is not None:
        np.save(os.path.join(results_dir, 'ensemble_labels.npy'), ensemble_labels)
    np.save(os.path.join(results_dir, 'true_labels.npy'), all_targets)

    print(f"Experiment complete. Results saved in {results_dir}")
    
    # Time measurement
    end = datetime.datetime.now(local_time_zone)
    delta = end-start
    total_seconds = int(delta.total_seconds())
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"Elapsed: {hours}h {minutes}m {seconds}s")


if __name__ == '__main__':
    main()
