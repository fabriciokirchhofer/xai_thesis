# Placeholder for strategy logic - to be extended later.

class StrategyFactory:
    @staticmethod
    def get_strategy(name):
        if name == 'average':
            return lambda preds: preds.mean(dim=0)
        raise NotImplementedError(f"Strategy {name} not implemented.")