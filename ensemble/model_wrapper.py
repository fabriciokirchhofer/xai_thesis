from ensemble.base_model import BaseModelXAI
from third_party import run_models  # uses get_model, load_checkpoint

class DenseNet121Model(BaseModelXAI):
    def _init_model(self):
        """Initialize the DenseNet121 model using existing factory logic."""
        self.model = run_models.get_model(model='DenseNet121', tasks=self.tasks, model_args=self.model_args)