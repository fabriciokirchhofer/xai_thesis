import torch
from torchvision import transforms
from abc import ABC, abstractmethod
from third_party import run_models, dataset, utils

args = run_models.parse_arguments()

class BaseModelXAI(ABC):
    def __init__(self, tasks:list = None, model_args=None):
        """
        Abstract base class for XAI-compatible models.

        Args:
            model_name (str): Name of the model architecture (e.g., DenseNet121).
            model_args: Parsed arguments from parse_arguments().
            tasks (list): List of target class names.
        """
        # 1) store the things *everyone* needs - to be defined with initialization
        self.model_args = model_args
        self.tasks = tasks
        

        # 2) declare placeholders for things subclasses must fill - those which are not defined yet
        self.model = None
        self.data_loader = None

        # 3) kick off the hooks:
        #    a) subclass must implement _initialize_model()
        #    b) then we load weights
        self._init_model()
        self._load_weights()


    @abstractmethod
    def _init_model(self):
        """Must be implemented by subclass to initialize architecture. Otherwise Error will be thrown."""
        pass

    def _load_weights(self):
        """Load checkpoint weights into model and set it to evaluation mode - NO training."""
        self.model = run_models.load_checkpoint(self.model, self.model_args.ckpt)
        self.model.eval()

    def prepare_data_loader(self, 
                            default_data_conditions:bool=True, 
                            batch_size_override:int=None, 
                            test_set:bool=False,
                            assign:bool=True):
        """
        Prepares a DataLoader based on model_args or optional override.
        Args:
            default_data_conditions (bool): By default it will use the data-loading conditions in run_models.py. 
            batch_size_override (int, optional): If specified, overrides args.batch_size.
            test_set (bool): Whether to load the test set.
            assign (bool): If True, assign loader to self.data_loader. If False the loader is just for temporary use e.g. to generate saliency maps. Will currently throw an error.
        Returns:
            DataLoader
        """
        if default_data_conditions:
            print(f"Loading data in BaseModel from default")
            loader = run_models.prepare_data(model_args=args)
        else:
            print(f"Run with overriden settings. See in model_cfg what was overriden.")
            batch_size = batch_size_override or self.model_args.batch_size
            
            if test_set:
                print("Get labels from Test set")
                labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'  
            else:
                labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/valid.csv'
            img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'

            transform = transforms.Compose([
                transforms.ConvertImageDtype(dtype=torch.float),
                transforms.Resize((320, 320)),
                transforms.Normalize(mean=[0.5032]*3, std=[0.2919]*3)
            ])

            loader = dataset.get_dataloader(
                annotations_file=labels_path,
                img_dir=img_path,
                transform=transform,
                batch_size=batch_size,
                test=test_set
            )
        if assign: # If false it will currently throw an error in run_class_model(self) -> not initialized!
            self.data_loader = loader
        return loader


    def run_class_model(self) -> torch.Tensor:
        """Runs the model on input images and returns logits."""
        if self.model is None or self.data_loader is None:
            raise ValueError("Model and data_loader must be initialized before calling run_class_model.")
        
        return run_models.model_run(model=self.model, data_loader=self.data_loader)


    def predict(self, images: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        """
        Runs the model on a batch of images and returns sigmoid probabilities.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device in BaseModelXAI for {args.model} is {device}")
        self.model.to(device)
        self.model.eval()
        images = images.to(device)

        with torch.no_grad():
            logits = self.model(images)
            if apply_sigmoid:
                return torch.sigmoid(logits)
            else:
                return logits


class DenseNet121Model(BaseModelXAI):
    def _init_model(self):
        """Initialize the DenseNet121 model using existing factory logic."""
        self.model = run_models.get_model(model='DenseNet121', tasks=self.tasks, model_args=self.model_args)

class ResNet152Model(BaseModelXAI):
    def _init_model(self):
        """Initialize the ResNet152 model using existing factory logic."""
        self.model = run_models.get_model(model='ResNet152', tasks=self.tasks, model_args=self.model_args)


def get_model_wrapper(model_name: str):
    """
    Returns the BaseModelXAI subclass object corresponding to the given architecture name.
    A wrapper that maps from the model name via args_model.model to the subclass object instantiater.
    """
    mapping = {
        'DenseNet121': DenseNet121Model,
        'ResNet152': ResNet152Model,
        # add other model wrappers here, for example:
        # 'ResNet101': ResNet101Model,
        # 'Inceptionv3': Inceptionv3Model,
    }
    try:
        return mapping[model_name]
    except KeyError:
        raise ValueError(f"Unknown model wrapper '{model_name}'. Available: {list(mapping.keys())}")
