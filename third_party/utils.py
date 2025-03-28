from sklearn import metrics
import numpy as np
import torch


#******************** Utils ********************

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

def auroc(predictions, ground_truth, tasks, n_classes=14):
    """
    Args:
        predictions (array like): of shape (n_samples,) or (n_samples, n_classes)
        ground_truth (array like): of shape (n_samples,) or (n_samples, n_classes)
        tasks (list of str): Names for each class/task.
        n_classes (int): must match the number of classes. By default 14
    
    Returns:
        dict: Mapping from task name to its ROC AUC score.

    """

    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
        #print(f"Shape of predictions from tensor: {predictions.shape}")
        #predictions = np.reshape(predictions,(-1,n_classes))
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
        #print(f"Shape of predictions from sth else: {predictions.shape}")
        #predictions = np.array(predictions).np.reshape(-1,n_classes)

    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()
        #print(f"Shape of ground_truth from tensor: {ground_truth.shape}")
        #ground_truth = np.reshape(ground_truth, (-1,n_classes))
    elif not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)
        #print(f"Shape of ground_truth from sth else: {ground_truth.shape}")
        #predictions = np.array(predictions).np.reshape(-1,n_classes)

    if len(tasks) != n_classes:
        raise ValueError(f"Expected {n_classes} tasks, but got {len(tasks)}")
    
    auc_results = {}
    for i, task in enumerate(tasks):
        gt = ground_truth[:, i]
        pred = predictions[:, i]
        
        # Check if there are both positive and negative samples.
        if len(np.unique(gt)) < 2:
            print(f"Warning: Only one class present in ground truth for {task}. Skipping ROC AUC computation.")
            auc_results[task] = float('nan')
        else:
            auc_results[task] = metrics.roc_auc_score(y_true=gt, y_score=pred)
    
    # auroc_score = metrics.roc_auc_score(y_true=ground_truth, 
    #                                     y_score=predictions,
    #                                     average='macro',
    #                                     multi_class='ovo') 

    return auc_results