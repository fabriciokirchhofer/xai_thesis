
import torch
import os
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

from third_party import parse_arguments, get_model, load_checkpoint, eval_model, prepare_data






# To access all the default arguments from run_models.pys
args = parse_arguments()

def load_model(model_name:str='DenseNet121', tasks:list=None, model_args=None)-> torch.nn.Module:
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


def get_target_layer(model:torch.nn.Module, layer_name:str=None):
    """
    Retrieve the target convolutional layer for Grad-CAM.

    Args:
        model (torch.nn.Module): Input model for which the specific layer shall be retrieved. By default DenseNet121.
        layer (str, optional): The layer which shall be retrieved. By default -1 to fit DenseNet121.

    Return:
        Specified layer of model for Grad-CAM.

    """
    # If a specific layer name is provided, try to use that:
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


def generate_gradcam_heatmap(model:torch.nn.Module, input_tensor:torch.Tensor, target_class:int, target_layer:torch.nn.Module) -> torch.Tensor:
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


def visualize_heatmap(heatmap, title:str = "Grad-CAM Heatmap", model_name:str='densenet121') -> None:
    plt.imshow(heatmap, cmap='jet')
    plt.title(title)
    plt.colorbar()
    # Return the current figure for further purposes 
    fig = plt.gcf()
    return fig

def save_heatmap(fig:plt.Figure, save_path:str) -> None:
    """
    Save the matplotlib figure to disk.
    
    Args:
        fig: The matplotlib figure to save.
        save_path: The full file path where to save the figure.
    """
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    


def main():

    print("\n\n************************ Get started with saliency maps script ************************\n\n")

    tasks = [
    'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion' , 'Edema' ,
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
    target_class_dict = {'Cardiomegaly':2,
                         'Edema':5,
                          'Consolidation':6,
                           'Atelectasis':8,
                            'Pleaural Effusion':10
                            }
    
    # Define output directory and ensure it exists.
    output_dir = os.path.expanduser('~/xai_thesis/saliency_maps')
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_name=args.model, tasks=tasks, model_args=args)
    print("Access layer")
    print("Type of model", type(model))
    layer = get_target_layer(model=model)

    data_loader = prepare_data(args)
    for img, label in data_loader:
        for target_name, idx in target_class_dict.items():
            print("Start generating maps")
            heatmap = generate_gradcam_heatmap(model=model, input_tensor=img, target_class=idx, target_layer=layer)
            fig = visualize_heatmap(heatmap=heatmap, title=(f"Grad-CM for {target_name}"))


            save_path = os.path.join(output_dir, f"{args.model}_{target_name}.png")
            print("Save path:", save_path)
            save_heatmap(fig, save_path)
            #plt.savefig('.~/xai_thesis/saliency_maps/' + str(args.model_name) + str(target_name) +'.png')

        break


    print("********** Finished saliency_maps script **********")
    return 0
    


if __name__ == "__main__":
    main()



