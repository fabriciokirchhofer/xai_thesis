
import torch
import numpy as np
import os

from torchvision import transforms
from abc import ABC, abstractmethod

from third_party import utils, run_models, dataset

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
        """Load checkpoint weights into model."""
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
            print(f"Loading data with customized settings.")
            batch_size = batch_size_override or self.model_args.batch_size
            
            if test_set:
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
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(device)
        # images = images.to(device)
        # with torch.no_grad():
        #     logits = self.model(images)
        # return logits

    # def predict(self, images: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
    #     """Returns probabilities by applying sigmoid to logits if specified."""
    #     logits = self.model_run(images)
    #     return torch.sigmoid(logits) if apply_sigmoid else logits

    # def compute_saliency(self, image: torch.Tensor, target_class: int, image_id: str) -> np.ndarray:
    #     """
    #     Retrieves or computes Grad-CAM saliency map for a given image and target class.

    #     Args:
    #         image (torch.Tensor): Image tensor of shape [1, C, H, W].
    #         target_class (int): Index of the class to compute saliency for.
    #         image_id (str): Unique identifier used for cache lookup.

    #     Returns:
    #         np.ndarray: Saliency heatmap.
    #     """
    #     model_type = self.model_args.model
    #     man_path = os.path.basename(os.path.dirname(self.model_args.ckpt))
    #     cache_dir = os.path.expanduser(f"~/repo/xai_thesis/heatmap_cache/{model_type}/{man_path}")
    #     os.makedirs(cache_dir, exist_ok=True)

    #     cache_path = os.path.join(cache_dir, f"{image_id}_{self.tasks[target_class]}.npz")
    #     if self.model_args.saliency == 'get' and os.path.exists(cache_path):
    #         return np.load(cache_path)['heatmap']

    #     layer = get_target_layer(self.model, self.gradcam_target_layer)
    #     heatmap = generate_gradcam_heatmap(self.model, image, target_class, layer)

    #     if self.model_args.saliency in ['compute', 'save_img']:
    #         np.savez_compressed(cache_path, heatmap=heatmap)

    #     return heatmap

    # def get_saliency_vector(self, heatmap: np.ndarray, size=(10, 10)) -> np.ndarray:
    #     """Processes heatmap into a flattened, normalized vector."""
    #     return process_heatmap(
    #         heatmap=heatmap,
    #         target_size=size,
    #         normalize=True,
    #         flatten=True,
    #         as_tensor=False
    #     )
