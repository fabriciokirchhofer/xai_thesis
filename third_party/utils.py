from sklearn import metrics
from sklearn.metrics import f1_score, auc, roc_curve
from sklearn.metrics import roc_auc_score
from captum.attr import LayerGradCam
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F



#***********************************************
#******************** Utils ********************
#***********************************************




#---------------------------------------------------
# ************* Generate Saliency maps *************
#---------------------------------------------------
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
    for idx, module in enumerate(model.modules()):
        #print(f"Module nr {idx+1} is {module}")
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer in the model.")
    #print(f"passed module to LayerGradCam is: {target_layer}")
    return target_layer # passed module for DenseNet121 is: Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


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
    # Convert heatmap to tensor with shape [1, 1, H, W]. Unsqueeze adds at dim 0 a tensor of size 1.
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)

    if normalize:
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



# -------------------------------------------------------------------
# *********** Utility functions for class distinctiveness ***********
# -------------------------------------------------------------------
def compute_centroids(saliency_dict: dict) -> dict:
    """
    Compute the mean (centroid) vector for each class.
    Args:
        saliency_dict: mapping class_name -> list of 1D torch.Tensor vectors
    Returns:
        centroids: mapping class_name -> 1D torch.Tensor (mean vector)
    """
    centroids = {}
    for cls, vecs in saliency_dict.items():
        stack = torch.stack(vecs, dim=0)  # shape: (N_c, D)
        centroids[cls] = stack.mean(dim=0)
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
        print("Entered plotting function")
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
            
            print(f"looping over {task}")
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
            fig.savefig('results/plots/roc_' + str(task) + '.png')
            
            # Add to combined plot
            ax_combined.plot(fpr, tpr, lw=2, label=f'{task} (AUC = {roc_auc:.2f})', linewidth=0.8)

        ax_combined.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
        ax_combined.set_xlim([0, 1])
        ax_combined.set_ylim([0, 1.05])
        ax_combined.set_xlabel('False Positive Rate')
        ax_combined.set_ylabel('True Positive Rate (sensitivity)')
        ax_combined.set_title('Combined ROC Curves for All Tasks')
        ax_combined.legend(loc="lower right", fontsize='small')
        fig_combined.savefig('results/plots/combined_roc.png')
