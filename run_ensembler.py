from ensemble import base_model

import argparse
from third_party.run_models import parse_arguments
from ensemble.model_wrapper import DenseNet121Model
from ensemble.ensemble import ModelEnsemble
from ensemble.evaluator import Evaluator

args = parse_arguments()

tasks = [
    'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion' , 'Edema' ,
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

model = DenseNet121Model(model_args=args, tasks=tasks)
data_loader = model.prepare_data_loader(default_data_conditions=True, batch_size_override=None, test_set=False)
print("Got loader")
logits = model.run_class_model()
print("Got logits")

ensemble = ModelEnsemble(models=[model], strategy='average')
evaluator = Evaluator(model_args=args, 
                      data_loader=data_loader, 
                      tasks=tasks, 
                      logits=logits,
                      only_max_prob_view=True)

evaluator.eval()
print("*************** run_ensembler script completed ***************")