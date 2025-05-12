from third_party import run_models
from ensemble import model_classes
from ensemble import ensemble
from ensemble import evaluator

args = run_models.parse_arguments()

tasks = [
    'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion' , 'Edema' ,
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

model = model_classes.DenseNet121Model(model_args=args, tasks=tasks)
data_loader = model.prepare_data_loader(default_data_conditions=True, batch_size_override=None, test_set=False)
print("Got loader")
logits = model.run_class_model()
print("Got logits")

ensemble = ensemble.ModelEnsemble(models=[model], strategy='average')

evaluator = evaluator.Evaluator(model_args=args, 
                      data_loader=data_loader, 
                      tasks=tasks, 
                      logits=logits,
                      only_max_prob_view=True)

evaluator.eval()
print("*************** run_ensembler script completed ***************")




# Each model can point to its own checkpoint and config via args.
# from ensemble.model_wrapper import DenseNet121Model, ResNet152Model

# model1 = DenseNet121Model(tasks=tasks, args=args1)
# model2 = ResNet152Model(tasks=tasks, args=args2)

# ensemble = ModelEnsemble(models=[model1, model2], strategy='average')
