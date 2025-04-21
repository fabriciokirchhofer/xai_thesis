import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from captum.attr import LayerGradCam

from third_party import parse_arguments, get_model, load_checkpoint, prepare_data


# To access all the default arguments from run_models.pys
args = parse_arguments()

def load_model(model_name:str='DenseNet121', tasks:list=None, model_args=None)->torch.nn.Module:
    """
    Load an initialize trained model in eval mode.
    Args:
        model_name (str): Either 'DenseNet121', 'ResNet152', 'Inceptionv4'. By default 'DenseNet121'.
        tasks (list): List with pathologies the model will look for and classify for its presence.
        model_args (various): See run_models.py
    Return:
        Initialized model in evaluation mode (torch.nn.Module).
    """
    # Use parsed arguments as default if none provided
    if model_args is None:
        model_args = args
        
    model = get_model(model_name, tasks, model_args)
    model = load_checkpoint(model, model_args.ckpt)
    return model


def get_target_layer(model:torch.nn.Module, layer_name:str=None)->torch.nn.Conv2d:
    """
    Retrieve the target convolutional layer for Grad-CAM.
    Args:
        model (torch.nn.Module): Input model for which the specific layer shall be retrieved. By default DenseNet121.
        layer (str, optional): The layer which shall be retrieved. By default -1 to fit DenseNet121.
    Return:
        Specified layer of model for Grad-CAM.
    """
    # If specific layer name (str) is provided, try to retrieve:
    if layer_name:
        try:
            target_layer = getattr(model, layer_name)
            return target_layer
        except AttributeError:
            print(f"Warning: Model has no attribute '{layer_name}'. Proceeding to auto-detect a convolutional layer.")

        # Auto-detect: Iterate through modules to find the last Conv2d layer.
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer in the model.")
    return target_layer


def generate_gradcam_heatmap(model:torch.nn.Module, input_tensor:torch.Tensor, target_class:int, target_layer:torch.nn.Module)->np.ndarray:
    """
    Generate a Grad-CAM heatmap for the specified input and target class.
    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): Input image tensor preprocessed as required (batch size of 1 assumed).
        target_class (int): Class index for which to compute the attribution.
        target_layer (torch.nn.Module): The layer to target for Grad-CAM.
    
    Returns:
        2D heatmap as a NumPy array.
    """
    grad_cam = LayerGradCam(model, target_layer)
    attributions = grad_cam.attribute(input_tensor, target=target_class)
    # Remove the batch dimension and convert to a NumPy array. If multi-channel, average them.
    heatmap = attributions.squeeze().detach().cpu().numpy()
    if heatmap.ndim == 3:
        heatmap = heatmap.mean(axis=0)
    return heatmap


def upscale_heatmap(heatmap, target_size:tuple=(320,320))->np.ndarray:
    """
    Upscale 2D heatmap to target size using bilinear interpolation.
    Args:
        heatmap: 2D numpy array e.g. 10x10 depnding on layer used to generate heatmap
        target_size: Tuple (height,width) for output resolution
    Return:
        Resized heatmap as numpy array.
    """
    # Convert heatmap to tensor with shape [1, 1, H, W]. Unsqueeze adds at dim 0 a tensor of size 1.
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    upscaled_heatmap = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return upscaled_heatmap.squeeze().numpy()


def overlay_heatmap_on_img(original_img:torch.tensor, heatmap:np.ndarray, alpha:float=0.3)->np.ndarray:
    """
    Overlay saliency map on original image.
    Args:
        original_img: Grayscale image as torch.tensor of size [batch_size, channels, hight, width]. Batch size must be 1 otherwise squeeze will not work
        heatmap: saliency map as numpy array of shape (hight, width)
        transparency: alpha value which matplotlib uses for blinding the saliency map
    Return:
        Overlay image as numpy array of shape (hight, width, channel).
    """
    # Normalize heatmap to [0, 1] using min-max scaling.
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # will have shape [height, width, 4] (RGBA). Take only first three channels.
    cmap = plt.get_cmap('jet')
    colored_heatmap = cmap(heatmap_norm)[:, :, :3]
    
    # Convert original image from a tensor of shape [batch_size=1, 3, height, width]
    # to a numpy array of shape [height, width, 3] and apply normalization
    img = original_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    overlay = (1 - alpha) * img + alpha * colored_heatmap
    
    # Clip final overlay to ensure the values are in [0, 1]. Avoid warning
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_heatmap(heatmap, title:str = "Grad-CAM Heatmap")->plt.figure:
    plt.imshow(heatmap, cmap='jet')
    plt.title(title)
    plt.colorbar()
    # Return the current figure for further purposes 
    fig = plt.gcf()
    return fig


def save_heatmap(fig:plt.Figure, save_path:str) -> None:
    """
    Save the matplotlib figure   
    Args:
        fig: The matplotlib figure to save.
        save_path: The full file path where to save the figure.
    """
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)   


def main():
    print("\n\n************************ Get started with saliency maps script ************************\n\n")

    tasks = [
    'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion' , 'Edema' ,
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
    target_class_dict = {
        'Cardiomegaly':2,
        'Edema':5,
        'Consolidation':6,
        'Atelectasis':8,
        'Pleaural Effusion':10
        }
    
    # Define output directory and ensure it exists.
    output_dir = os.path.expanduser('~/repo/xai_thesis/saliency_maps/')
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_name=args.model, tasks=tasks, model_args=args)
    layer = get_target_layer(model=model)

    data_loader = prepare_data(args)
    for img, _ in data_loader:
        print("Start generating maps")
        for target_name, idx in target_class_dict.items():
            heatmap = generate_gradcam_heatmap(model=model, input_tensor=img, target_class=idx, target_layer=layer)
            
            # Process and save GradCam overlay to original img.
            upscaled_heatmap = upscale_heatmap(heatmap=heatmap, target_size=(320,320))
            overlayed_imgs = overlay_heatmap_on_img(original_img=img, heatmap=upscaled_heatmap, alpha=0.4)
            fig = visualize_heatmap(heatmap=overlayed_imgs, title=(f"Grad-CM for {target_name}"))
            save_path = os.path.join(output_dir, f"{args.model}/{target_name}.png")
            print("Save path:", save_path)
            save_heatmap(fig, save_path)

    print("********** Finished saliency_maps script **********")
    


if __name__ == "__main__":
    main()



