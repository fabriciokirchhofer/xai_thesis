import torch
import numpy as np
import json
import pandas as pd
from ensemble import evaluator

class ModelEnsemble:
    def __init__(self, models, strategy='average', **strategy_params):
        self.models = models
        self.strategy_fn = StrategyFactory.get_strategy(strategy, **strategy_params)

    def predict_batch(self, images):
        preds = [m.predict(images).cpu() for m in self.models]
        stack = torch.stack(preds, dim=0)
        try:
            return self.strategy_fn(stack)
        except TypeError:
            return self.strategy_fn([p.numpy() for p in preds])

    def predict_loader(self, data_loader):
        all_preds, all_targets = [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used in prediciton loader: %s", device)
        for images, labels in data_loader:
            images = images.to(device)
            batch_preds = self.predict_batch(images)
            if isinstance(batch_preds, torch.Tensor):
                all_preds.append(batch_preds.cpu())
            else:
                all_preds.append(torch.from_numpy(batch_preds))
            all_targets.append(labels)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_preds, all_targets


def _mean_dist(preds_list):
    arr = np.stack(preds_list, axis=-1)
    mean = np.mean(arr, axis=-1, keepdims=True)
    dist = np.abs(arr - mean) + 1e-8
    inv_dist = 1.0 / dist
    weights = inv_dist / np.sum(inv_dist, axis=-1, keepdims=True)
    return np.sum(arr * weights, axis=-1)   

class StrategyFactory:
    @staticmethod
    def get_strategy(name, **params):
        name = name.lower()

        # Simple average of probabilities
        if name == 'average':
            def avg_fn(preds, all_targets=None):
                # preds: list of tensors or numpy arrays, or tensor
                if isinstance(preds, list):
                    # if torch tensors, stack directly
                    if torch.is_tensor(preds[0]):
                        stack = torch.stack(preds, dim=0)
                        return stack.mean(dim=0).numpy()
                    else:
                        arr = np.stack(preds, axis=0)
                        return np.mean(arr, axis=0)
                elif torch.is_tensor(preds):
                    return preds.mean(dim=0).numpy()
                else:
                    arr = np.stack(preds, axis=0)
                    return np.mean(arr, axis=0)
            return avg_fn

        # Weighted average with fixed weights
        if name == 'weighted':
            weights = params.get('weights') # TODO: Define 'weights' in the config file. In run_experiments.py it will be called as ensemble_cfg (ca line 102)
            assert weights is not None, "Weights must be provided for weighted strategy"
            weights = np.array(weights, dtype=float)
            def w_fn(preds, all_targets=None):
                # preds: list of tensors or numpy arrays
                if isinstance(preds, list) and torch.is_tensor(preds[0]):
                    stack = torch.stack(preds, dim=0).numpy()
                elif isinstance(preds, list):
                    stack = np.stack(preds, axis=0)
                else:
                    stack = np.stack(preds, axis=0)
                # Normalize weights
                w = weights / weights.sum()
                # Weighted sum across model axis (axis=0)
                combined = np.tensordot(stack, w, axes=([0], [0]))
                return combined
            return w_fn

        # Model weight based on distance from mean
        if name == 'mean_distance_weighted':
            return lambda preds: _mean_dist(preds)

        # Majority voting thresholded at vote_threshold
        if name == 'voting':
            thresh = params.get('vote_threshold', None)
            if thresh == None:
                print("No thresholds passed - will take default 0.5")
                thresh = 0.5
            def v_fn(preds, all_targets=None):
                # preds: list of tensors or numpy arrays
                if isinstance(preds, list) and torch.is_tensor(preds[0]):
                    arr = torch.stack(preds, dim=0).numpy()
                else:
                    arr = np.stack(preds, axis=0)
                    print(f"Thresholds: {thresh}")
                votes = (arr >= thresh).astype(int)
                maj = (votes.sum(axis=0) > (arr.shape[0] / 2)).astype(float)
                return maj
            return v_fn
        
        if name == 'distinctiveness_weighted':
            # Load distinctiveness values for each model
            distinct_files = params.get('distinctiveness_files')
            distinct_vals_list = []
            if distinct_files:
                # Load JSON distinctiveness for each model
                for file_path in distinct_files:
                    with open(file_path, 'r') as f:
                        distinct_vals_list.append(json.load(f))
                print(f"These is the list with the dist values: {distinct_vals_list}")
            elif params.get('distinctiveness_values'):
                # Accept distinctiveness values directly (list of dicts or lists)
                print("Got em from list")
                distinct_vals_list = params['distinctiveness_values']
            else:
                raise ValueError("Distinctiveness data not provided. Please specify 'distinctiveness_files' or "
                                 "'distinctiveness_values'")

            # Determine class ordering (for mapping class names to prediction indices)
            # If a full task list or class name list is provided, use it; otherwise assume CheXpert default order.
            class_names = params.get('class_names') or params.get('tasks')
            if class_names:
                print(f"Got class names from config file: {class_names}")
                tasks_list = class_names
            else:
                print("Use default tasks.")
                tasks_list = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                        'Lung Opacity', 'Lung Lesion', 'Edema',
                        'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                        'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

            num_models = len(distinct_vals_list)
            num_classes = len(tasks_list)
            # Initialize weight matrix (models x classes) with ones. Default value which will be used for normalization.
            weight_matrix = np.ones((num_models, num_classes), dtype=float)

            
            for i, dist_dict in enumerate(distinct_vals_list):
                # Fix possible key mismatch (e.g., "Pleaural Effusion" typo)
                if 'Pleaural Effusion' in dist_dict:
                    dist_dict['Pleural Effusion'] = dist_dict.pop('Pleaural Effusion')
                    print(f"Corrected typo in tasks.")
                for cls_name, dist_val in dist_dict.items():
                    if cls_name in tasks_list:
                        j = tasks_list.index(cls_name)
                        weight_matrix[i, j] = dist_val
                    else:
                        print(f"the class name: {cls_name} is not in the task list. Go back and double check!")
            # Normalize weights across models for each class (so columns sum to 1)
            weight_matrix = weight_matrix / weight_matrix.sum(axis=0, keepdims=True)
            #weight_matrix = 1 / weight_matrix # Inversion for ablation
            print(f"Weight matrix after normalizaiton: {weight_matrix}")

            # Ensemble function using the computed weights
            def distinctiveness_fn(preds, all_targets=None):
                # Convert predictions list/tensor to a NumPy array of shape (M, N, C)
                if isinstance(preds, list):
                    if torch.is_tensor(preds[0]):
                        stack = torch.stack(preds, dim=0).numpy()  # shape: (models, N, C)
                    else:
                        stack = np.stack(preds, axis=0)            # shape: (models, N, C)
                elif torch.is_tensor(preds):
                    stack = preds.numpy()  # already stacked tensor of shape (models, N, C)
                else:
                    stack = np.stack(preds, axis=0)


                # Compute weighted sum across model axis (axis=0) using the weight matrix
                # Expand weight_matrix to shape (models, 1, C) for broadcasting across N samples
                print(f"Weighted matrix extended: {weight_matrix[:, np.newaxis, :].shape}")
                weighted_sum = np.sum(stack * weight_matrix[:, np.newaxis, :], axis=0)
                print(f"Shape of weighted_sum: {weighted_sum.shape}")
                return weighted_sum  # shape: (N, C) NumPy array            
            # make it accessible
            distinctiveness_fn.weight_matrix = weight_matrix
            return distinctiveness_fn
        

        if name == 'distinctiveness_voting':
            """
            Threshold is computed first for every model and then after for the ensemble which is distinctiveness weighted.
            The ensemble returns a voting fraction for every class per image. For the ensemble the threshold is recomputed.
            Threshold is set by default to 0.5.
            """
            # Load per-model distinctiveness dictionaries
            distinct_vals_list = []
            if 'distinctiveness_files' in params:
                for path in params['distinctiveness_files']:
                    with open(path, 'r') as f:
                        distinct_vals_list.append(json.load(f))
            elif 'distinctiveness_values' in params:
                distinct_vals_list = params['distinctiveness_values']
            else:
                raise ValueError("Distinctiveness data required for voting: provide 'distinctiveness_files' or 'distinctiveness_values'.")

            # Determine class order
            tasks_list = params.get('class_names') or params.get('tasks')
            if tasks_list is None:
                tasks_list = [
                    'No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                    'Lung Opacity','Lung Lesion','Edema','Consolidation',
                    'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
                    'Pleural Other','Fracture','Support Devices'
                ]

            num_models = len(distinct_vals_list)
            num_classes = len(tasks_list)

            weight_matrix = np.ones((num_models, num_classes), dtype=float)
            for i, dist in enumerate(distinct_vals_list):
                # fix common typo
                if 'Pleaural Effusion' in dist:
                    dist['Pleural Effusion'] = dist.pop('Pleaural Effusion')
                    print("Corrected typo in tasks. Correct it manually in the json file!")
                for cls, val in dist.items():
                    if cls in tasks_list:
                        j = tasks_list.index(cls)
                        weight_matrix[i, j] = val
            # Normalize per-class so sum of model weights = 1
            weight_matrix /= weight_matrix.sum(axis=0, keepdims=True)

            
            # Capture validation targets and threshold config
            tune_cfg = params.get('threshold_tuning', {})
            tuning_stage = tune_cfg.get('stage', 'none')
            metric = tune_cfg.get('metric', 'f1')

            def dv_soft(preds, all_targets):
                # preds: list of tensors or arrays, or stacked tensor
                if isinstance(preds, list):
                    if torch.is_tensor(preds[0]):
                        stack = torch.stack(preds, dim=0).cpu().numpy()
                    else:
                        stack = np.stack(preds, axis=0)
                elif torch.is_tensor(preds):
                    stack = preds.cpu().numpy()
                else:
                    stack = np.array(preds)

                if tuning_stage == 'none':
                    print("Based on config params to Test mode for labels and treshold retrival")
                    test = True
                    per_model_voting_thresholds_path = params['per_model_voting_thresholds_path']
                    per_model_thresholds = np.load(per_model_voting_thresholds_path, allow_pickle=True)
                else: 
                    print("Based on config params to Val mode for labels retrival and threshold creation")
                    test = False
                    thresholds = None

                votes_list = []
                threshold_arrays = []
                # Loop over all models to get for each its maximum probability per study view
                for idx, model_preds in enumerate(stack):
                    votes, gt_labels = _get_max_prob_per_view(model_preds, all_targets, tasks_list, args=test)
                    votes_list.append(votes)
                    if not test:
                        thresholds = evaluator.find_optimal_thresholds(probabilities=votes_list[-1], 
                                                                   ground_truth=gt_labels,
                                                                   tasks=tasks_list,
                                                                   metric=metric)[0]
                        arr = np.array([thresholds[cls] for cls in tasks_list], dtype=float)
                        threshold_arrays.append(arr)
                        #thresholds_list.append(thresholds) # Before putting thresholds to array -> Fixed
                    else:
                        thresholds = per_model_thresholds[idx]
                        threshold_arrays.append(thresholds)                   
                    # Compute threshold based labels
                    if thresholds is not None:
                        votes_list[-1]  = evaluator.threshold_based_predictions(probs=torch.tensor(votes_list[-1]),
                                                                                     thresholds=thresholds,
                                                                                     tasks=tasks_list).numpy()
                    else:
                        print("No threshold tuning applied. Will take default threshold 0.5 for each model")
                        votes_list[-1]  = (votes_list[-1] >= 0.5).astype(float)

                votes_arr = np.stack(votes_list, axis=0)
                per_model_voting_thresholds = np.stack(threshold_arrays, axis=0)

                # Compute weighted vote fractions
                # weight_matrix: (M, C) -> expand to (M, 1, C)
                weighted = votes_arr * weight_matrix[:, np.newaxis, :]
                soft_scores = weighted.sum(axis=0)
                return soft_scores, gt_labels, per_model_voting_thresholds  # shape (N, C), in [0,1]
            return dv_soft
        

        if name == 'average_voting':
            """
            The models receive each equal weight. Threshold is computed first for every model and then after for the ensemble.
            Threshold is computed for each class and is set by default to 0.5.
            """
            # Determine class order
            tasks_list = params.get('class_names') or params.get('tasks')
            if tasks_list is None:
                tasks_list = [
                    'No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                    'Lung Opacity','Lung Lesion','Edema','Consolidation',
                    'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
                    'Pleural Other','Fracture','Support Devices'
                ]
            
            # Capture validation targets and threshold config
            tune_cfg = params.get('threshold_tuning', {})
            tuning_stage = tune_cfg.get('stage', 'none')
            metric = tune_cfg.get('metric', 'f1')

            def av_soft(preds, all_targets):
                # preds: list of tensors or arrays, or stacked tensor
                if isinstance(preds, list):
                    if torch.is_tensor(preds[0]):
                        stack = torch.stack(preds, dim=0).cpu().numpy()
                    else:
                        stack = np.stack(preds, axis=0)
                elif torch.is_tensor(preds):
                    stack = preds.cpu().numpy()
                else:
                    stack = np.array(preds)

                
                if tuning_stage == 'none':
                    print("Based on config params to Test mode for labels and treshold retrival")
                    test = True
                    per_model_voting_thresholds_path = params['per_model_voting_thresholds_path']
                    per_model_thresholds = np.load(per_model_voting_thresholds_path, allow_pickle=True)
                else: 
                    print("Based on config params to Val mode for labels retrival and threshold creation")
                    test = False
                    thresholds = None

                votes_list = []
                threshold_arrays = []
                # Loop over all models to get for each its maximum probability per study view
                for idx, model_preds in enumerate(stack):
                    votes, gt_labels = _get_max_prob_per_view(model_preds, all_targets, tasks_list, args=test)
                    votes_list.append(votes)
                    if not test:
                        thresholds = evaluator.find_optimal_thresholds(probabilities=votes_list[-1], 
                                                                   ground_truth=gt_labels,
                                                                   tasks=tasks_list,
                                                                   metric=metric)[0]
                        arr = np.array([thresholds[cls] for cls in tasks_list], dtype=float)
                        threshold_arrays.append(arr)
                        #thresholds_list.append(thresholds) # Before putting thresholds to array -> Fixed
                    else:
                        thresholds = per_model_thresholds[idx]
                        threshold_arrays.append(thresholds)                   
                    # Compute threshold based labels
                    if thresholds is not None:
                        votes_list[-1]  = evaluator.threshold_based_predictions(probs=torch.tensor(votes_list[-1]),
                                                                                     thresholds=thresholds,
                                                                                     tasks=tasks_list).numpy()
                    else:
                        print("No threshold tuning applied. Will take default threshold 0.5 for each model")
                        votes_list[-1]  = (votes_list[-1] >= 0.5).astype(float)

                votes_arr = np.stack(votes_list, axis=0)
                avg_votes = np.mean(votes_arr, axis=0)
                #avg_votes = 1/((avg_votes - avg_votes.min()) / (avg_votes.max()-avg_votes.min() + 1e-6) + 1e-6) # Invert for ablation study
                per_model_voting_thresholds = np.stack(threshold_arrays, axis=0)

                return avg_votes, gt_labels, per_model_voting_thresholds  # shape (N, C), in [0,1]
            return av_soft
        

# *****************************************************
# ****************** Helper funtions ******************
# *****************************************************

def _extract_study_id(mode):
    """
    Args:
        mode (bool): either False goes to 'val' and True goes to 'test'. 
            It will then read the corresponding CSV file and extract the study ID
            from the image file paths (assuming the study ID is encoded in the first
            three parts of the file path, separated by '/').
    Returns:
        DataFrame: The CSV DataFrame with an added 'study_id' column.
    """
    if mode == False:
        df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/valid.csv')
        # Apply lambda to the first column to extract the study id.
        df['study_id'] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[:3]))
        return df

    elif mode:
        df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/test.csv')
        df['study_id'] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[:3]))
        return df

    else:
        raise ValueError("Expected either 'val' or 'test' mode, but got something else.")

def _get_max_prob_per_view(probs, gt_labels, tasks: list = None, args=None):
    # 1) bring inputs into plain numpy floats
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()

    # 2) load study IDs
    df = _extract_study_id(mode=args)

    # 3) build DataFrames
    prob_df = pd.DataFrame(probs, columns=tasks)
    gt_df   = pd.DataFrame(gt_labels, columns=tasks)
    prob_df['study_id'] = df['study_id']
    gt_df['study_id']   = df['study_id']

    # 4) group and take max per study
    agg_prob = prob_df.groupby('study_id').max()
    agg_gt   = gt_df.groupby('study_id').max()
    #print(f"Current aggregated ground truth df: {agg_gt.head()}")

    # 5) convert back to torch tensors of floats
    prob_arr = agg_prob.to_numpy(dtype=np.float32)
    gt_arr   = agg_gt.to_numpy(dtype=np.float32)

    return prob_arr, gt_arr