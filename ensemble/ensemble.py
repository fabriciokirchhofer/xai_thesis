import torch
import numpy as np
import json

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
            def avg_fn(preds):
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
            weights = params.get('weights')
            assert weights is not None, "Weights must be provided for weighted strategy"
            weights = np.array(weights, dtype=float)
            def w_fn(preds):
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
            thresh = params.get('vote_threshold', 0.5)
            def v_fn(preds):
                # preds: list of tensors or numpy arrays
                if isinstance(preds, list) and torch.is_tensor(preds[0]):
                    arr = torch.stack(preds, dim=0).numpy()
                else:
                    arr = np.stack(preds, axis=0)
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

            # Fill weight matrix using inverse distinctiveness for known classes
            for i, dist_dict in enumerate(distinct_vals_list):
                # Fix possible key mismatch (e.g., "Pleaural Effusion" typo)
                if 'Pleaural Effusion' in dist_dict:
                    dist_dict['Pleural Effusion'] = dist_dict.pop('Pleaural Effusion')
                    print(f"Corrected typo in tasks.")
                for cls_name, dist_val in dist_dict.items():
                    if cls_name in tasks_list:
                        j = tasks_list.index(cls_name)
                        # Inverse distinctiveness as weight (add tiny epsilon to avoid div-by-zero)
                        weight_matrix[i, j] = 1.0 / (dist_val + 1e-8)
                    else:
                        print(f"the class name: {cls_name} is not in the task list. Go back and double check!")
            print(f"Weight matrix befroe normalization: {weight_matrix}")
            # Normalize weights across models for each class (so columns sum to 1)
            weight_matrix = weight_matrix / weight_matrix.sum(axis=0, keepdims=True)
            #weight_matrix = 1 / weight_matrix
            print(f"Weight matrix after normalizaiton: {weight_matrix}")

            # Define the ensemble function using the computed weights
            def distinctiveness_fn(preds):
                # Convert predictions list/tensor to a NumPy array of shape (M, N, C)
                if isinstance(preds, list):
                    print(f"The len of preds-list is {len(preds)}")
                    if torch.is_tensor(preds[0]):
                        print(f"The first element of the list is a tensor of the size: {preds[0].size()}")
                        stack = torch.stack(preds, dim=0).numpy()   # shape: (models, N, C)
                        print(f"Type {type(stack)}")
                        print(f"The stacked torch tensor predictions along the first dim are now of shape: {stack.shape}")
                    else:
                        stack = np.stack(preds, axis=0)            # shape: (models, N, C)
                        print(f"The stacked numpy predictions along the first dim are now of shape: {stack.size()}")
                elif torch.is_tensor(preds):
                    print(f"Predictions are a torch tensor of size: {preds.size()}")
                    stack = preds.numpy()  # already stacked tensor of shape (models, N, C)
                else:
                    print(f"Preds are already a numpy array of shape: {preds.shape()}")
                    stack = np.stack(preds, axis=0)
                    print(f"Preds are now after stacking a numpy array of shape: {preds.shape()}")

                # Compute weighted sum across model axis (axis=0) using the weight matrix
                # Expand weight_matrix to shape (models, 1, C) for broadcasting across N samples
                print(f"Weighted matrix extended: {weight_matrix[:, np.newaxis, :].shape}")
                weighted_sum = np.sum(stack * weight_matrix[:, np.newaxis, :], axis=0)
                print(f"Shape of weighted_sum: {weighted_sum.shape}")
                return weighted_sum  # shape: (N, C) NumPy array

            return distinctiveness_fn
