from sklearn import metrics
from sklearn.metrics import f1_score, auc, roc_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
from captum.attr import LayerGradCam, LayerLRP
import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


#---------------------------------------------------
# ************* Generate Saliency maps *************
#---------------------------------------------------
def get_target_layer(model:torch.nn.Module, layer_name:str=None, method:str='gradcam')->torch.nn.Conv2d:
    """
    Retrieve the target convolutional layer for Grad-CAM.
    Args:
        model (torch.nn.Module): Input model for which the specific layer shall be retrieved.
        layer (str, optional): The layer which shall be retrieved. By default -1 to fit DenseNet121.
        method (str, optional): The saliency method applied to access the relevant layer. By default GradCAM -> Last conv layer
    Return:
        Specified layer of model for saliency method.
    """
    # If specific layer name (str) is provided, try to retrieve:
    if layer_name:
        try:
            target_layer = getattr(model, layer_name)
            return target_layer
        except AttributeError:
            print(f"Warning: Model has no attribute '{layer_name}'. Proceeding to auto-detect a convolutional layer.")

    # Auto-detect: Iterate through modules to find layer
    target_layer = None
    for idx, module in enumerate(model.modules()):
        #print(f"Module nr {idx+1} is {module}")
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            ## Section for first conv layer with ResNet152 and LRP
            # if model.__class__.__name__ == 'DenseNet121' and method=='lrp':
            #     print(f"Idx when returning layer for LRP: {idx}")
            #     print("Recognized Resnet")
            #     return target_layer
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer in the model.")
    return target_layer


def generate_gradcam_heatmap(model:torch.nn.Module, 
                             input_tensor:torch.Tensor, 
                             target_class:int, 
                             target_layer:torch.nn.Module)->np.ndarray:
    """
    Generate a Grad-CAM heatmap for the specified input and target class.
    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): Input image tensor preprocessed as required. Batch size of 1 assumed, otherwise it will run but behave wrongly.
        target_class (int): Class index for which to compute the attribution.
        target_layer (torch.nn.Module): By default the last conv layer for Grad-CAM.
    Returns:
        2D heatmap as a NumPy array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    target_layer = target_layer.to(device)


    grad_cam = LayerGradCam(forward_func=model, layer=target_layer)
    attributions = grad_cam.attribute(input_tensor, target=target_class)
    # Remove the batch dimension and convert to a NumPy array. If multi-channel, average them.
    heatmap = attributions.squeeze().detach().cpu().numpy()
    if heatmap.ndim == 3:
        heatmap = heatmap.mean(axis=0)
    return heatmap


def generate_lrp_attribution(model:torch.nn.Module,
                                    input_tensor:torch.Tensor,
                                    target_class:int,
                                    target_layer:torch.nn.Module) -> np.ndarray:
    """
    Generate a LRP attribution map for a given input and target class.
    Args:
        model (torch.nn.Module): Classification network in eval() mode. Must use ReLU activations.
        input_tensor (torch.Tensor): A single preprocessed input image of shape [1, C, H, W].
        target_class (int): Index of the output neuron (class) for which to compute attributions.
        target_layer (torch.nn.Module): By default the last conv layer for Grad-CAM.
    Returns:
        TBD    

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    target_layer = target_layer.to(device)

    from captum.attr._utils.lrp_rules import EpsilonRule
    model.eval() 
    input_tensor = input_tensor.requires_grad_(True)
    # Apply the Epsilon LRP rule to all Conv2D and Linear layers in the model
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            module.rule = EpsilonRule()  # attach Epsilon rule for stability in LRP
    
    # Initialize Captum LRP on the entire model (propagates relevance from output to input)
    lrp = LayerLRP(model=model, layer=target_layer)
    # Compute LRP attributions for the specified target class.
    # This returns a tensor with the same shape as the input (e.g., 1 x 3 x H x W for an image).
    attributions = lrp.attribute(inputs=input_tensor, 
                                 target=target_class, 
                                 attribute_to_layer_input=True, 
                                 verbose=False)
    # print(f"Shape of attribution: {attributions.size()}")
    # print(f"The attribution: {attributions}")

    attr = attributions.squeeze().detach().cpu().numpy()

    # Aggregate across channels if needed
    if attr.ndim == 3:
        attr = attr.mean(axis=0)

    return attr


def ig_heatmap(model:torch.nn.Module,
                input_tensor:torch.Tensor,
                target_class:int,
                target_layer:torch.nn.Module) -> np.ndarray:
    
    from captum.attr import IntegratedGradients
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Cuda device name for IG: {torch.cuda.get_device_name()}")
    # input_tensor = input_tensor.to(device)
    #model = model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ig = IntegratedGradients(model)
    ig_attrib, delta = ig.attribute(input_tensor, 
                             target=target_class, 
                             baselines=torch.zeros_like(input_tensor).to(device),
                             n_steps=200, # Smallest convergence delta with 80
                             internal_batch_size=1,
                             return_convergence_delta=True)
    #print(f"Returned convergence delta is: {delta} and device name: {torch.cuda.get_device_name()}")
    heatmap_ig = ig_attrib[0].mean(dim=0).cpu().numpy()
    # print(f"Shape of attributions: {heatmap_ig.shape}")
    # print(f"IG attributions: {heatmap_ig}")

    # Aggregate across channels if needed [C,M,N] -> [M,N]
    if heatmap_ig.ndim == 3:
        heatmap_ig = heatmap_ig.mean(axis=0)

    return heatmap_ig



from captum.attr import LayerDeepLift

def deep_lift_layer_heatmap(model, layer, input_tensor, target_class, baseline=None):
    model = model.eval().to(input_tensor.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor, device=input_tensor.device)

    ldl = LayerDeepLift(model=model, layer=layer)
    attr = ldl.attribute(inputs=input_tensor,
                         baselines=baseline,
                         target=target_class)
    # attr shape: [1, C, H, W]
    heatmap = attr.squeeze().cpu().detach().numpy().mean(axis=0)
    # Aggregate over channels
    if heatmap.ndim == 3:
        heatmap = heatmap.mean(axis=0)
    return heatmap


# Verion only working with DenseNet121 not with ResNet152
# from captum.attr import DeepLift
# def deep_lift_heatmap(model: torch.nn.Module,
#                       input_tensor: torch.Tensor,
#                       target_class: int,
#                       baseline: torch.Tensor = None) -> np.ndarray:
#     """
#     Generate a DeepLIFT attribution map for the specified input and target class.
#     Args:
#         model (torch.nn.Module): The trained model in eval mode.
#         input_tensor (torch.Tensor): Preprocessed input image of shape [1,C,H,W].
#         target_class (int): Index of the class for which to compute attributions.
#         baseline (torch.Tensor, optional): Reference input of same shape as input_tensor.
#             Defaults to all-zero baseline.
#     Returns:
#         2D heatmap as a NumPy array.
#     """
#     import torch.nn as nn
#     model = model.eval()
#     for m in model.modules():
#         if hasattr(m, "inplace"):
#             m.inplace = False

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     input_tensor = input_tensor.to(device)
#     # Default baseline = zero tensor if none provided
#     if baseline is None:
#         baseline = torch.zeros_like(input_tensor).to(device)

#     # Compute DeepLIFT attributions
#     dl = DeepLift(model)
#     attributions = dl.attribute(inputs=input_tensor,
#                                 baselines=baseline,
#                                 target=target_class)
#     # attributions shape: [1, C, H, W]
#     heatmap = attributions.squeeze().detach().cpu().numpy()
#     # Aggregate over channels
#     if heatmap.ndim == 3:
#         heatmap = heatmap.mean(axis=0)
#     return heatmap


def process_heatmap(heatmap:np.ndarray, 
                    target_size:tuple=(10, 10), 
                    normalize:bool=False, 
                    flatten:bool=False, 
                    as_tensor:bool=False)->np.ndarray:
    """
    Upscale 2D heatmap to target size using bilinear interpolation.
    Args:
        heatmap (2D numpy array): Heatmap to be processed. E.g. 10x10 depnding on layer used to generate heatmap
        target_size (Tuple): (height,width) for output resolution
        normalize (bool): If True max-min normalization will be performed to move heatmap values into range of [0,1]. By default False.
        flatten (bool): If True flattening operation will be performed to obtain vectorized latent layer heatmap. By default False.
        as_tensor (bool): If True heatmap will be returnes as tensor, otherwise by default as numpy array.
    Return:
        Resized heatmap as numpy array.
    """
    tolerance = 1e-6
    # Convert heatmap to tensor with shape [1, 1, H, W]. Unsqueeze adds at dim 0 a tensor of size 1.
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    #print(f"Target size: {target_size}\nheatmap.size: {heatmap[0, 0, :, :].size()}")
    if target_size != heatmap[0, 0, :, :].size():
        heatmap = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)
        #print("Interpolated heatmap")

    if normalize:
        if heatmap.max()-heatmap.min() < tolerance:
            print("Uniform heatmap detected: skipping normalization.")
            return None
        heatmap = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min() + 1e-8)
    if flatten:
        heatmap = torch.flatten(heatmap)
    if as_tensor:
        return heatmap
    else:
        return heatmap.squeeze().numpy()    


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
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # will have shape [height, width, 4] (RGBA). Take only first three channels.
    cmap = plt.get_cmap('jet')
    colored_heatmap = cmap(heatmap_norm)[:, :, :3]
    
    # Convert original image from a tensor of shape [batch_size=1, 3, height, width]
    # to a numpy array of shape [height, width, 3] and apply normalization
    img = original_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

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

# -------------------------------------------------------------------
# *********** Utility functions for class distinctiveness ***********
# -------------------------------------------------------------------
def class_distinctiveness(saliency_dict:dict, function:str='cosine_similarity')->dict:
    """
    Function to calculate pair-wise class distinctiveness based on cosine similarity
    Args:
        saliency_dict (dict): Dictionary in structure of dict[str, list[np.ndarray]] containing class_names and flattened heatmaps.
    Return:
        distinctiveness (dict): Pair-wise class distinctiveness in structure of dict[tuple(str,str), float32] containing pair-wise class names and dist. value.
    """
    class_names = list(saliency_dict.keys())
    stacked_vectors = np.stack([np.array(saliency_dict[name]).ravel() for name in class_names], axis=0)
    #print(f"***\nShape of stacked vectors: {stacked_vectors.shape}\ntype of stacked vectors {type(stacked_vectors)}")
    if function=='cosine_similarity':
        # 1-cos_sim = distinctiveness
        similarity_matrix = 1 - cosine_similarity(stacked_vectors)
    elif function=='cosine_distance':
        #print(f"Compute cosine distance of stacked vectors: {stacked_vectors}")
        # defined as 1.0 minus the cosine similarity -> range 0 to 2. 0=equal, 2=diametral opposite
        similarity_matrix = cosine_distances(stacked_vectors)
    else:
        raise ValueError("Expected either 'cosine_similarity' or 'cosine_distances' as measuring metric, but got something else.")

    distinctiveness = {}
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            pair = (class_names[i], class_names[j])
            distinctiveness[pair] = similarity_matrix[i, j]
    return distinctiveness

def sum_up_distinctiveness(collection:dict, new_scores)->dict:
    """
    Helper function to gather pair-wise distinctiveness of several images in dictionary.
    Args: 
        collection (dict): Pair-wise class distinctiveness in structure of dict[tuple(str,str), float32] containing pair-wise class names and dist. value.
        new_scores (dict): Pair-wise class distinctiveness in structure of dict[tuple(str,str), float32] containing pair-wise class names and dist. value.
    Return:
        collection (dict): Collection of pair-wise class distinctiveness in structure of dict[tuple(str,str), list[float32]] containing pair-wise class names and list of cosine-similarity values.
    """
    for pair, val in new_scores.items():
        # If key named pair already exists it returns collection[pair], if not yet existing it creates new key with empty list.
        collection.setdefault(pair, []).append(val)
    return collection


def per_class_distinctiveness(distinctiveness_collection:dict,
                              normalize:bool=True,
                              save_path:str=None,
                              ckpt_name:str=None)->dict:
    """
    Plots a boxplot for each class’s distinctiveness distribution and optionally saves it,
    and computes a per-class distinctiveness score.

    Args:
        distinctiveness_collection: dict[tuple(str,str) -> list[float]].
            Pairwise class distinctiveness values.
        normalize: whether to min–max normalize before plotting.
        save_path: directory where outputs will be saved.
            If provided, it'll save:
              - class_wise_boxplot.png
              - class_wise_distinctiveness.json

    Returns:
        class_wise_distinctiveness: dict[str -> float]
            Mapping from class name to its mean distinctiveness.
            If save_path is given, returns 0 after saving.
    """
    # 1) raw or normalized data
    data = normalize and normalize_distinctiveness(distinctiveness_collection) \
           or distinctiveness_collection

    # 2) gather all values per class
    class_values = {}
    for (a, b), vals in data.items():
        class_values.setdefault(a, []).extend(vals)
        class_values.setdefault(b, []).extend(vals)

    # 3) prepare for boxplot
    classes = sorted(class_values.keys())
    values = [class_values[cls] for cls in classes]

    plt.figure(figsize=(max(6, len(classes)*0.5), 6))
    plt.boxplot(values, labels=classes, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Distinctiveness")
    plt.title("Class-wise Distinctiveness")
    plt.tight_layout()

    # 4) compute mean distinctiveness per class
    class_wise_distinctiveness = {
        cls: float(np.mean(class_values[cls])) if class_values[cls] else float("nan")
        for cls in classes
    }

    # 5) save if requested
    if save_path:
        plot_path = os.path.join(save_path, ckpt_name + "_class_wise_boxplot.png")
        plt.savefig(plot_path)
        # JSON of means
        json_path = os.path.join(save_path, ckpt_name + "_class_wise_distinctiveness.json")
        with open(json_path, "w") as f:
            json.dump(class_wise_distinctiveness, f, indent=4)
        plt.close()
        return 0
    else:
        print(f"No path provided. Nothing to be stored."
              f"Class_wise_distinctiveness: {class_wise_distinctiveness}")
        return class_wise_distinctiveness
    

def normalize_distinctiveness(distinctiveness_collection: dict)->dict:
    """
    Min–max normalizes all values across every class-pair in the collection.
    Args:
        distinctiveness_collection: dict[tuple(str,str), list[float]]
            List of raw distinctiveness scores for each class-pair.
    Returns:
        dict[tuple(str,str), list[float]] of the same shape, where each score
        has been transformed to (x - global_min)/(global_max - global_min).
        If all values are equal or collection is empty, returns zeros or empty.
    """
    all_vals = []
    for vals in distinctiveness_collection.values():
        all_vals.extend(vals)
    min_val, max_val = min(all_vals), max(all_vals)
    span = max_val -min_val

    normalized = {}
    for pair, vals in distinctiveness_collection.items():
        if span > 0:
            normalized[pair] = [(v - min_val) / span for v in vals]
        else:
            # avoid division by zero if all values are identical
            normalized[pair] = [0.0 for _ in vals]
    return normalized


def plot_distinctiveness_boxplots(distinctiveness_collection: dict,
                                  normalize: bool = False,
                                  save_path: str = None)->plt.Figure:
    """
    Plots a boxplot for each class-pair’s distinctiveness distribution and optionally saves it.
    Args:
        distinctiveness_collection: dict[tuple(str,str) -> list[float]]
        normalize (bool): whether to min–max normalize before plotting.
        save_path: full file path where the figure will be saved.
            If provided, the directory will be created if needed.
    Returns:
        The matplotlib Figure object for further customization or saving.
    """
    # choose raw or normalized data
    data = normalize and normalize_distinctiveness(distinctiveness_collection) or distinctiveness_collection

    pairs = list(data.keys())
    values = [data[pair] for pair in pairs]

    fig, ax = plt.subplots()
    ax.boxplot(values, labels=[f"{a}-{b}" for a, b in pairs])
    title = "Normalized Distinctiveness" if normalize else "Distinctiveness"
    ax.set_title(f"{title} per Class Pair")
    ax.set_xlabel("Class Pair")
    ax.set_ylabel("Value")
    ax.grid(visible=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    # Save if requested
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=300)





# In each loop it stacks the vectors of each cls together (class-wise) 
# and then calculates the mean and inserts it into the dictionary centroids (class-wise)
def compute_centroids(saliency_dict: dict) -> dict:
    """
    Compute the mean (centroid) vector for each class.
    Args:
        saliency_dict: mapping class_name -> list of 1D torch.Tensor vectors
    Returns:
        centroids: mapping class_name -> 1D torch.Tensor (mean vector)
    """
    centroids = {}
    cnt = 0
    for cls, vecs in saliency_dict.items():     
        stack = torch.stack(vecs, dim=0)  # shape: (N_c, D) Number of saliency maps (234) and Dimension of vector (100)
        centroids[cls] = stack.mean(dim=0)
        print(f"Stack round {cnt}.\nstack size:{stack.size()}\ncentroids:{centroids}")
        cnt += 1
    return centroids

def compute_distinctiveness(centroids: dict) -> dict:
    """
    Compute per-class distinctiveness scores via cosine similarity.
    D(c) = 1 - avg_{c' != c} cos(mu_c, mu_c').
    """
    class_names = list(centroids.keys())
    C = len(class_names)
    mat = torch.stack([centroids[c] for c in class_names], dim=0)
    mat_norm = mat / (mat.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = mat_norm @ mat_norm.t()  # shape (C, C)

    distinctiveness = {}
    for i, c in enumerate(class_names):
        others = [j for j in range(C) if j != i]
        mean_sim = cos_sim[i, others].mean().item()
        distinctiveness[c] = 1.0 - mean_sim
    return distinctiveness


def plot_distinctiveness(distinctiveness: dict, save_path: str = None) -> plt.Figure:
    """
    Plot a bar chart of per-class distinctiveness scores.
    """
    classes = list(distinctiveness.keys())
    values = [distinctiveness[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(classes, values)
    ax.set_ylabel('Distinctiveness')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_title('Per-Class Distinctiveness')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# --------------------------------------------------------------
#***************** Preprocessing and inference *****************
# --------------------------------------------------------------
def remove_prefix(dict, prefix):
    """
    Function to remove additional prefix created while saving the model
    should be sth like: "model.features.norm0.weight" but is currently "model.model.features.norm0.weight"
    """
    new_state_dict = {}
    for key, value in dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def extract_study_id(mode):
    """
    Args:
        mode (bool): either False goes to 'val' and True goes to 'test'. 
            It will then read the corresponding CSV file and extract the study ID
            from the image file paths (assuming the study ID is encoded in the first
            three parts of the file path, separated by '/').
    Returns:
        DataFrame: The CSV DataFrame with an added 'study_id' column.
    """
    if mode == False:
        df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/valid.csv')
        # Apply lambda to the first column to extract the study id.
        df['study_id'] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[:3]))
        return df

    elif mode:
        df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/test.csv')
        df['study_id'] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[:3]))
        return df

    else:
        raise ValueError("Expected either 'val' or 'test' mode, but got something else.")


def compute_accuracy(predictions, labels):
    """
    Computes overall binary accuracy across all tasks.
    """
    correct = (predictions == labels).float().sum()
    total = labels.numel()
    accuracy = correct / total
    return accuracy.item()

def get_max_prob_per_view(probs, gt_labels, tasks: list = None, args=None):
    # 1) bring inputs into plain numpy floats
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()

    # 2) load study IDs
    df = extract_study_id(mode=args.run_test)

    # 3) build DataFrames
    prob_df = pd.DataFrame(probs, columns=tasks)
    gt_df   = pd.DataFrame(gt_labels, columns=tasks)
    prob_df['study_id'] = df['study_id']
    gt_df['study_id']   = df['study_id']

    # 4) group and take max per study
    agg_prob = prob_df.groupby('study_id').max()
    agg_gt   = gt_df.groupby('study_id').max()
    #print(f"Current aggregated ground truth df: {agg_gt.head()}")

    # 5) convert back to torch tensors of floats
    prob_arr = agg_prob.to_numpy(dtype=np.float32)
    gt_arr   = agg_gt.to_numpy(dtype=np.float32)

    return prob_arr, gt_arr


def comput_youden_idx(ground_truth, preds, tasks):
    youden_idx = {}
    for i, task in enumerate(tasks):
        pred = preds[:, i]
        # Compute confusion matrix components
        tp = ((ground_truth[:, i] == 1) & (pred == 1)).sum()
        fn = ((ground_truth[:, i] == 1) & (pred == 0)).sum()
        tn = ((ground_truth[:, i] == 0) & (pred == 0)).sum()
        fp = ((ground_truth[:, i] == 0) & (pred == 1)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = sensitivity + specificity - 1
        youden_idx[task] = score
    return youden_idx

def compute_f1_score(ground_truth, preds, tasks):
    f1_scores = {}
    for i, task in enumerate(tasks):
        pred = preds[:, i]
        score = f1_score(ground_truth[:, i], pred, zero_division=0.0)
        f1_scores[task] = score
    return f1_scores

def auroc(probabilities:np.ndarray, ground_truth:np.ndarray, tasks, n_classes=14):
    """
    Computes per-class AUROC using continuous probabilities.
    Args:
        predictions (array like): of shape (n_samples,) or (n_samples, n_classes)
        ground_truth (array like): of shape (n_samples,) or (n_samples, n_classes)
        tasks (list of str): Names for each class/task.
        n_classes (int): must match the number of classes. By default 14
    
    Returns:
        dict: Mapping from task name to its ROC AUC score.

    """
    # Check format of predictions
    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()
    elif not isinstance(probabilities, np.ndarray):       
        probabilities = np.array(probabilities).reshape(-1,n_classes)
        #probabilities = np.array(probabilities)

    # Check format of gt labels
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()
    elif not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth).reshape(-1,n_classes)

    # Catch mismatch of classes and tasks (model dependent)
    if len(tasks) != n_classes:
        raise ValueError(f"Expected {n_classes} tasks, but got {len(tasks)}")
    
    auc_results = {}
    for i, task in enumerate(tasks):
        gt = ground_truth[:, i]
        pred = probabilities[:, i]
        
        # Check if there are both positive and negative samples.
        # Here imbalances show up (e.g. fractures)
        if len(np.unique(gt)) < 2:
            print(f"Warning: Only one class present in ground truth for {task}. Skipping ROC AUC computation.")
            auc_results[task] = float('nan')
        else:
            auc_results[task] = roc_auc_score(y_true=gt, y_score=pred)
    return auc_results

def bootstrap_auc(y_true, y_score, n_bootstraps=1000):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    n = y_true.shape[0]
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = roc_auc_score(y_true[idx], y_score[idx])
        bootstrapped_scores.append(score)
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    return lower, upper


def find_optimal_thresholds(probabilities, ground_truth, tasks, step=0.01, metric="f1"):
    """
    Finds optimal threshold for each class based on maximizing the F1 score.

    Args:
        probabilities (numpy array): Array of shape (n_samples, n_classes) containing probabilities.
        ground_truth (numpy array): Array of shape (n_samples, n_classes) with true binary labels.
        tasks (list of str): List of class names.
        step (float): Step size for threshold search.

    Returns:
        optimal_thresholds (dict): Mapping from task to optimal threshold.
        metric_score_dict (dict): Mapping from task to the achieved score ().
    """
    optimal_thresholds = {}
    metric_score_dict = {}
    thresholds = np.arange(0, 1 + step, step)
    #n_classes = probabilities.shape[1]
    for i, task in enumerate(tasks):
        best_metric_score = -float("inf")
        best_threshold = 0.5  # default if nothing better is found
        for t in thresholds:
            preds = (probabilities[:, i] >= t).astype(int) # Bolean value (0,1) will become integer

            if metric == "f1":
                # F1-score = harmonic_mean(precision, sensitivity)=2TP/(2TP+FP+FN))
                score = f1_score(ground_truth[:, i], preds, zero_division=0.0) # If current score better than others -> overwrite
            elif metric == "youden":
                # Compute confusion matrix components
                tp = ((ground_truth[:, i] == 1) & (preds == 1)).sum()
                fn = ((ground_truth[:, i] == 1) & (preds == 0)).sum()
                tn = ((ground_truth[:, i] == 0) & (preds == 0)).sum()
                fp = ((ground_truth[:, i] == 0) & (preds == 1)).sum()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sensitivity + specificity - 1
            else:
                raise ValueError("Invalid metric selected. Must be either 'f1' or 'youden'.")           
            if score > best_metric_score:
                best_metric_score = score
                best_threshold = t
        optimal_thresholds[task] = best_threshold
        metric_score_dict[task] = best_metric_score
    return optimal_thresholds, metric_score_dict


def threshold_based_predictions(probs, thresholds, tasks):
    """
    Applies task-specific thresholds to probabilities to generate binary predictions.  
    Args:
        probs (Tensor): shape (N, C), predicted probabilities
        thresholds (float or dict): single threshold or per-task threshold dict
        tasks (List[str]): list of task names
    
    Returns:
        Tensor: binary predictions (0.0 or 1.0), shape (N, C)
    """
    if isinstance(thresholds, dict):
        predictions = torch.zeros_like(probs)
        for i, task in enumerate(tasks):
            predictions[:, i] = (probs[:, i] >= thresholds[task]).float()
    else:
        predictions = (probs >= thresholds).float()
    return predictions


# -------------------------------------------------------------------------
# ******************************* ROC PLOTS *******************************
# -------------------------------------------------------------------------
def plot_roc(predictions, ground_truth, tasks, n_classes=14):
        """
        Computes per-class AUROC using continuous probabilities.
        Args:
            predictions (array like): of shape (n_samples,) or (n_samples, n_classes)
            ground_truth (array like): of shape (n_samples,) or (n_samples, n_classes)
            tasks (list of str): Names for each class/task.
            n_classes (int): must match the number of classes. By default 14
        """
        # Convert to numpy arrays if necessary
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.detach().cpu().numpy()
            
        if len(tasks) != n_classes:
            raise ValueError(f"Expected {n_classes} tasks, but got {len(tasks)}")

        # Create a figure for the combined ROC curves
        fig_combined, ax_combined = plt.subplots(figsize=(8, 6))

        for i, task in enumerate(tasks):
            gt = ground_truth[:, i]
            pred = predictions[:, i]
            
            # Skip tasks where only one class is present in the ground truth
            if len(np.unique(gt)) < 2:
                print(f"Warning: Only one class present in ground truth for {task}. Skipping ROC curve plot for this task.")
                continue
            
            # calculate FP and TP rates
            fpr, tpr, _ = roc_curve(gt, pred)
            roc_auc = auc(fpr, tpr)
            
            # Plot individual ROC curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve for {task}')
            ax.legend(loc="lower right")
            fig.savefig('results/roc_plots/roc_' + str(task) + '.png')
            
            # Add to combined plot
            ax_combined.plot(fpr, tpr, lw=2, label=f'{task} (AUC = {roc_auc:.2f})', linewidth=0.8)

        ax_combined.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
        ax_combined.set_xlim([0, 1])
        ax_combined.set_ylim([0, 1.05])
        ax_combined.set_xlabel('False Positive Rate')
        ax_combined.set_ylabel('True Positive Rate (sensitivity)')
        ax_combined.set_title('Combined ROC Curves for All Tasks')
        ax_combined.legend(loc="lower right", fontsize='small')
        fig_combined.savefig('results/roc_plots/combined_roc.png')




# -------------------------------------------------------------------------
# ******************************* Confusion Matrix PLOTS *******************************
# -------------------------------------------------------------------------

def plot_CM(predictions, gt_labels, tasks, model_name):

    save_dir = "results/confusion_matrices"
    save_dir = os.path.join(save_dir, model_name)
    print(f"The directory where the CM will be stored: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # convert preds to numpy
    preds_np = predictions.numpy()

    # “viridis”, “plasma”, “inferno”
    cmap = 'viridis'

    # 1) Per‐task confusion matrices (as before)
    for i, task in enumerate(tasks):
        cm = confusion_matrix(gt_labels[:, i], preds_np[:, i])
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(f'Confusion Matrix: {task}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Neg', 'Pos'])
        ax.set_yticklabels(['Neg', 'Pos'])
        for (j, k), v in np.ndenumerate(cm):
            ax.text(k, j, v, ha='center', va='center')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{task.replace(' ', '_')}_cm.png"), dpi=150)
        plt.close(fig)

    # 2) One “overall” confusion‐matrix table for *all* tasks
    #    we build a DataFrame with rows=tasks, cols=[TN, FP, FN, TP]
    rows = []
    for i, task in enumerate(tasks):
        tn, fp, fn, tp = confusion_matrix(gt_labels[:, i], preds_np[:, i]).ravel()
        rows.append([tn, fp, fn, tp])

    df_cm = pd.DataFrame(rows,
                        index=tasks,
                        columns=['TN', 'FP', 'FN', 'TP'])

    # save raw numbers
    df_cm.to_csv(os.path.join(save_dir, "overall_confusion_table.csv"))

    # plot as a single heatmap‐style figure
    fig, ax = plt.subplots(figsize=(8, 3 + len(tasks)*0.3))
    im = ax.imshow(df_cm.values, interpolation='nearest', aspect='auto', cmap=cmap)
    ax.set_title("Per-Task Confusion Matrix Counts")
    ax.set_xticks(np.arange(df_cm.shape[1]))
    ax.set_yticks(np.arange(df_cm.shape[0]))
    ax.set_xticklabels(df_cm.columns)
    ax.set_yticklabels(df_cm.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for (j, k), v in np.ndenumerate(df_cm.values):
        ax.text(k, j, int(v), ha='center', va='center',
                color='white' if df_cm.values.max() > 50 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "overall_confusion_table.png"), dpi=150)
    plt.close(fig)



def plot_distinctiveness_heatmap_from_files(distinctiveness_files,
                                            models,
                                            cmap: str = 'viridis',
                                            figsize: tuple = (8, 6),
                                            vmin=None,
                                            vmax=None,
                                            save_path: str = None):
    """
    Load per-class distinctiveness from JSON files and plot a heatmap.

    Parameters
    ----------
    distinctiveness_files : list of str
        Paths to JSON files. Each must map class_name -> distinctiveness value.
    models : list of str
        Model identifiers, in the same order as distinctiveness_files.
    cmap : str or Colormap, optional
        Matplotlib colormap.
    figsize : tuple, optional
        Figure size.
    vmin, vmax : float, optional
        Color scale limits.
    save_path : str, optional
        If provided, save the figure here at 300 dpi (dirs auto‑created).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # --- load JSONs into a matrix
    class_names = None
    matrix = []
    for fp in distinctiveness_files:
        with open(fp, 'r') as f:
            data = json.load(f)
        if class_names is None:
            # preserve the JSON's key order if possible
            class_names = list(data.keys())
        else:
            if set(data.keys()) != set(class_names):
                raise ValueError(f"Class names in {fp} differ from previous files")
        matrix.append([data[c] for c in class_names])
    D = np.array(matrix)

    # --- plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(D, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Models')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Distinctiveness')
    plt.tight_layout()

    # --- save if requested
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        combined_path = os.path.join(save_path, 'distinctiveness_heatmap.png')
        fig.savefig(combined_path, dpi=300)
        plt.close(fig)



def plot_distinctiveness_radar_from_files(distinctiveness_files,
                                          models,
                                          figsize: tuple = (6, 6),
                                          fill_alpha: float = 0.1,
                                          line_width: float = 1.5,
                                          save_path: str = None):
    """
    Load per-class distinctiveness from JSON files and plot a radar chart.

    Parameters
    ----------
    distinctiveness_files : list of str
        Paths to JSON files. Each must map class_name -> distinctiveness value.
    models : list of str
        Model identifiers, in the same order as distinctiveness_files.
    figsize : tuple, optional
        Figure size.
    fill_alpha : float, optional
        Radar fill opacity.
    line_width : float, optional
        Radar line width.
    save_path : str, optional
        If provided, save the figure here at 300 dpi (dirs auto‑created).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # --- load JSONs
    class_names = None
    matrix = []
    for fp in distinctiveness_files:
        with open(fp, 'r') as f:
            data = json.load(f)
        if class_names is None:
            class_names = list(data.keys())
        else:
            if set(data.keys()) != set(class_names):
                raise ValueError(f"Class names in {fp} differ from previous files")
        matrix.append([data[c] for c in class_names])
    D = np.array(matrix)

    # --- compute angles
    n_classes = len(class_names)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
    angles += angles[:1]

    # --- plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i, m in enumerate(models):
        vals = D[i].tolist() + [D[i][0]]
        ax.plot(angles, vals, linewidth=line_width, label=m)
        ax.fill(angles, vals, alpha=fill_alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, max(1, D.max()))
    ax.set_yticks([])  # hide radial ticks
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    # --- save if requested
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        combined_path = os.path.join(save_path, 'distinctiveness_spider.png')
        fig.savefig(combined_path, dpi=300)
        plt.close(fig)




