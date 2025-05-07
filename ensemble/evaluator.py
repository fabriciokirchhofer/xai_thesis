#from sklearn.metrics import f1_score, roc_auc_score
from third_party.run_models import eval_model 
import numpy as np

class Evaluator:
    def __init__(self, model_args, data_loader, tasks, logits, only_max_prob_view:bool=True):
        self.model_args = model_args
        self.data_loader = data_loader
        self.tasks = tasks
        self.logits = logits
        self.only_max_prob_view = only_max_prob_view

    def eval(self):
        return eval_model(
            self.model_args, 
            self.data_loader, 
            self.tasks, 
            self.logits, 
            self.only_max_prob_view)