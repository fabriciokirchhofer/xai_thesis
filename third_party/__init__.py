# third_party/__init__.py

# Import specific functions to make them available directly from the package
from third_party.run_models import parse_arguments, prepare_data, get_model, load_checkpoint, eval_model
from third_party.utils import remove_prefix, extract_study_id, compute_accuracy, comput_youden_idx, compute_f1_score

# Define package-level variables
__version__ = '0.1.0'
__author__ = 'Fabricio'

# You could also run initialization code here
#print("Initializing third_party package")