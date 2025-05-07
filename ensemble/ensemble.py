import torch

class ModelEnsemble:
    def __init__(self, models, strategy='average'):
        self.models = models
        self.strategy = strategy
        self.weights = None

    def predict(self, images):
        preds = [m.predict(images) for m in self.models]
        stack = torch.stack(preds)

        if self.strategy == 'average':
            return stack.mean(dim=0)
        elif self.strategy == 'weighted' and self.weights:
            weights = torch.tensor(self.weights).view(-1, 1, 1)
            return (stack * weights).sum(dim=0)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")