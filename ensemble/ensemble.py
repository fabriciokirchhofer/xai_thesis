import torch
import numpy as np

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



