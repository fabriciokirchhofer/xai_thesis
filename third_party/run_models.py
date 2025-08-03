import argparse
import os
import torch
from torchvision import transforms
import json
# python -m third_party.run_models
import third_party.utils as utils
import third_party.dataset as dataset
import third_party.models as models

# import utils
# import dataset
# import models

# from third_party import utils
# from third_party import dataset
# from third_party import models

import csv
import pandas as pd



with open("config.json", "r") as f:
    config = json.load(f)
requested_device = config.get("device", "cuda:0")
if requested_device.startswith("cuda") and not torch.cuda.is_available():
    DEVICE = torch.device("cpu")
    print("No GPU available")
else:
    DEVICE = torch.device(requested_device)
    print(f"Connected to {torch.cuda.get_device_name(DEVICE)}")

path_dir = os.path.expanduser('~/repo/xai_thesis/third_party/pretrainedmodels/')
# Calculated on validation set average AUROC over atelectasis, cardiomegaly, consolidation, edema, and pleural effusion
# DenseNet121
ckpt_d_ignore_1 = os.path.join(path_dir, 'densenet121/uncertainty/densenet_ignore_1/epoch=2-chexpert_competition_AUROC=0.87_v1.ckpt') # AUROC = 0.8245074203947084 # torch.Size([14, 1024])
ckpt_d_ignore_2 = os.path.join(path_dir, 'densenet121/uncertainty/densenet_ignore_2/epoch=2-chexpert_competition_AUROC=0.88.ckpt') # AUROC = 0.8560705112383094
ckpt_d_ignore_3 = os.path.join(path_dir, 'densenet121/uncertainty/densenet_ignore_3/epoch=2-chexpert_competition_AUROC=0.88.ckpt') # AUROC = 0.854155053128039

# Resnet152
ckpt_r_ignore_2 = os.path.join(path_dir, 'resnet152/resnet_ignore_2/epoch=2-chexpert_competition_AUROC=0.86.ckpt') # AUROC = 0.7969021549350594
ckpt_r_ignore_3_ep2_1 = os.path.join(path_dir, 'resnet152/resnet_ignore_3/epoch=2-chexpert_competition_AUROC=0.86.ckpt') # AUROC = 0.8141815614997918
ckpt_r_ignore_3x_ep2_2 = os.path.join(path_dir, 'resnet152/resnet_ignore_3/epoch=2-chexpert_competition_AUROC=0.87.ckpt') # AUROC = 0.8207077659558035
ckpt_r_ignore_3x_ep1 = os.path.join(path_dir, 'resnet152/resnet_ignore_3/epoch=1-chexpert_competition_AUROC=0.87.ckpt') # AUROC = 0.8258944682967464

# Inception V4
ckpt_i_ignore_1 = os.path.join(path_dir, 'inceptionv4/inception_ignore_1/epoch=2-chexpert_competition_AUROC=0.85.ckpt') # AUROC = 0.4681767038450782 (with inception preprocessing otherwise AUROC = 0.4972145074342835)
ckpt_i_ignore_2 = os.path.join(path_dir, 'inceptionv4/inception_ignore_2/epoch=2-chexpert_competition_AUROC=0.86_v2.ckpt') # AUROC =  0.47889381692531796 (with inception preprocessing otherwise AUROC = 0.4492819192735242)
ckpt_i_ignore_3 = os.path.join(path_dir, 'inceptionv4/inception_ignore_3/epoch=2-chexpert_competition_AUROC=0.85.ckpt') # AUROC = 0.4928516042460034 (with inception preprocessing otherwise  AUROC = 0.41869342379579316)


# Parse arguments -> Argumente Zerlegung
def create_parser():
    parser = argparse.ArgumentParser(description="Script settings to run XAI model ensemble")
    parser.add_argument('--pretrained',type=bool, default=True, help='Use pre-trained model')
    parser.add_argument('--model_uncertainty', type=bool, default=False, help='Use model uncertainty') # If not further used it can be removed
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size which will be passed to the model')
    parser.add_argument('--model', type=str, default='DenseNet121', help='specify model name')
    parser.add_argument('--ckpt', type=str, default=ckpt_d_ignore_1, help='Path to checkpoint file')

    parser.add_argument('--save_acc_roc', type=bool, default=False, help='Save accuracy and auroc during validation to csv file')
    parser.add_argument('--sigmoid_threshold', type=float, default=0.5, help='The threshold to activate sigmoid function. Used for model evaluation in validation.')
    parser.add_argument('--tune_thresholds', type=bool, default=True, help='If True, find optimal per-class thresholds using F1 score. Will save it.')
    parser.add_argument('--metric', type=str, default='f1', help='Choose evaluation evaluation metric. Can be "f1" or "youden".')   
    parser.add_argument('--run_test', type=bool, default=False, help='Runs the test set for evaluation. Needs thresholds from tune_thresholds as a csv file.')

    parser.add_argument('--plot_roc', type=bool, default=False, help='Plot the ROC curves for each task. Default false.')
    parser.add_argument('--saliency', type=str, default='compute', help='Whether to compute and save="compute", retreive stored="get", or compute and save imgage_maps="save_img"')
    return parser

# Thin wrapper to take arguments from outside
def parse_arguments():
    parser = create_parser()
    return parser.parse_args()

def get_model(model:str, tasks:list, model_args):
    """
    Factory to return an XAI model by name.
    Args:
        model_name (str): Name of the model architecture.
        tasks (list):     List of target class names.
        model_args: Parsed arguments namespace.
    Returns:
        An instance of the requested BaseModelXAI subclass.
    Raises:
        ValueError: If `model_name` is not recognized.
    """
    # Mapping to choose the right model class
    model_map = {
        'DenseNet121': models.DenseNet121,
        'ResNet152': models.ResNet152,
        'Inceptionv4': models.Inceptionv4
    }

    key = model
    if key not in model_map:
        valid = ', '.join(sorted(model_map.keys()))
        raise ValueError(
            f"Unknown model '{model}'."
            f"Please choose one of: {valid}"
        )

    model_class = model_map[key]
    return model_class(tasks=tasks, model_args=model_args)


# Load checkpoint and its parameters - original one
def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = utils.remove_prefix(state_dict, "model.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Prep dataset
def prepare_data(model_args):

    train_mean = torch.tensor([0.5032, 0.5032, 0.5032])
    train_std = torch.tensor([0.2919, 0.2919, 0.2919])
    size = (320, 320)

    # if model_args.model == 'Inceptionv4':
    #     print("Overwrite stats for Inceptionv4 model")
    #     # mean and std for normalization based on requirements for Inception v4
    #     train_mean = torch.tensor([0.5, 0.5, 0.5])
    #     train_std = torch.tensor([0.5, 0.5, 0.5])
    #     size = (299, 299)

    # Define inference transformation pipeline for the inception v4 architecture
    inference_transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize(size), # Resizing based of requirements for Inception v4
        transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())                
        ])
    
    if not model_args.run_test:
        print("Prepare validation data...") 
        data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/valid.csv'
        data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'

        data_loader = dataset.get_dataloader(
            annotations_file=data_labels_path,
            img_dir=data_img_path,
            transform=inference_transform,
            batch_size=model_args.batch_size,
            shuffle=False,
            test=False
            )

    if model_args.run_test:
        print("Prepare test data...")
        # Hardcoded normalization parameters from test set (could also be computed but takes some time to run)
        #test_mean = torch.tensor([128.0847, 128.0847, 128.0847])/255
        #test_std = torch.tensor([74.5220, 74.5220, 74.5220])/255

        data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'
        data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0'

        data_loader = dataset.get_dataloader(
            annotations_file=data_labels_path,
            img_dir=data_img_path,
            transform=inference_transform,
            batch_size=model_args.batch_size,
            shuffle=False,
            test=True
        )

    return data_loader


# Run the model
def model_run(model, data_loader):
    # Use GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Device for evaluation:", device)
    model.to(DEVICE)
    model.eval()

    print("Running model in standard multi-label mode...\n")
    all_logits = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE)
            logits = model(images)
            all_logits.append(logits.cpu())

    # Concatenate batch results from list into one torch tensor
    all_logits = torch.cat(all_logits, dim=0)

    return all_logits


def eval_model(model_args, data_loader, tasks, logits, only_max_prob_view:bool=True):
    """
    TODO: define args and returns
    Evaluate the model's performance using logits and ground truth from data_loader. 
    Computes accuracy, AUROC, optionally plots ROC curves, and performs threshold tuning. 

    """

    # Gather ground truth labels from data_loader size(n_images, tasks)
    gt_labels = []
    for images, labels in data_loader:
        gt_labels.append(labels.cpu())
    gt_labels = torch.cat(gt_labels, dim=0)
    print(f"*******************GT labels sanity check in run_models.py\n Tensor shape is {gt_labels.shape}")
    numpy_arr = gt_labels.detach().numpy()
    df = pd.DataFrame(numpy_arr)
    df_tasks = df.iloc[:, :]
    counts = df_tasks.sum()
    print(f"Shape of Dataframe: {df.shape}")
    print(counts)
    print("**********************************************End of GT labels sanity check")




    print("****Start evaluation mode****")
    probs = torch.sigmoid(logits).cpu()

    if only_max_prob_view:
        print(f"Get per patient only view with highest probability.")
        #******************** get max prob per view start ********************
        df = utils.extract_study_id(mode=model_args.run_test)
        #print("Current df:", df.head())

        # Convert probs and gt_labels to df
        prob_df = pd.DataFrame(probs.detach().cpu().numpy(), columns=tasks)
        gt_df   = pd.DataFrame(gt_labels.detach().cpu().numpy(), columns=tasks)

        # Match each row of predictions and gt to the df from the csv file
        prob_df['study_id'] = df['study_id']
        gt_df['study_id'] = df['study_id']

        # Group by study_id and take only the maximum predicted probability per study.
        agg_prob = prob_df.groupby('study_id').max()
        agg_gt = gt_df.groupby('study_id').max()
    
        probs = torch.tensor(agg_prob.values)
        gt_labels   = torch.tensor(agg_gt.values)
        #******************** get max prob per view end ********************

    # VALIDATION SET
    if not model_args.run_test:
        predictions = utils.threshold_based_predictions(probs, model_args.sigmoid_threshold, tasks)
        
        probs = probs.cpu()
        predictions = predictions.cpu()

        for i in range(5):
            print(f"\nSample {i}:\n  GT = {gt_labels[i].tolist()},\nPred = {[round(float(p),3) for p in probs[i]]}")

        acc = utils.compute_accuracy(predictions, gt_labels)
        print("-" * 40)
        print(f"Overall Accuracy with default threshold {model_args.sigmoid_threshold}: {acc:.4f}")

        auroc = utils.auroc(probabilities=probs, ground_truth=gt_labels, tasks=tasks)
        print("-" * 40)
        print(f"AUROC from sigmoid based probabilities:")
        eval_tasks = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        eval_auroc = 0
        for task, score in auroc.items(): 
            print(f"{task}: score: {score:.4f}")
            if task in eval_tasks:
                eval_auroc += score
        eval_auroc /= 5
        print('Final evaluation AUROC:', eval_auroc)
    
        if model_args.plot_roc:
            print("Plot the ROC curves")
            utils.plot_roc(predictions=probs, ground_truth=gt_labels, tasks=tasks)    

        # Compute optimal per-class thresholds based on F1 score, if threshold tuning is enabled.
        if model_args.tune_thresholds:
            probs_np = probs.numpy()
            labels_np = gt_labels.numpy()
            optimal_thresholds, score = utils.find_optimal_thresholds(probabilities=probs_np, ground_truth=labels_np, tasks=tasks, metric=model_args.metric)
            print("-" * 40)
            print(f"\nOptimal thresholds per class based on {model_args.metric} score:")
            for task in tasks:
                print(f"{task}: threshold = {optimal_thresholds[task]:.2f}, {model_args.metric} score = {score[task]:.4f}")
            # Apply the tuned thresholds to generate new predictions and compute accuracy

            tuned_predictions = torch.zeros_like(predictions)
            for i, task in enumerate(tasks):
                tuned_predictions[:, i] = (probs[:, i] >= optimal_thresholds[task]).float()
            tuned_acc = utils.compute_accuracy(tuned_predictions, gt_labels)
            print(f"Overall Accuracy with tuned thresholds: {tuned_acc:.4f}")


            eval_tasks = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
            target_class_dict = {
                'Cardiomegaly':2,
                'Edema':5,
                'Consolidation':6,
                'Atelectasis':8,
                'Pleural Effusion':10
                }    
                    
            print(f"Shape of tuned_predictions: {tuned_predictions.shape}")
            print(f"Shape of gt_labels: {gt_labels.shape}")
            print(f"target_class_dict.values: {target_class_dict.values()}")
            indices = list(target_class_dict.values())
            tuned_predictions_subset = tuned_predictions[:, indices]
            gt_labels_subset = gt_labels[:, indices]
            print(f"Shape of tuned_predictions_subset: {tuned_predictions_subset.shape}")
            print(f"Shape of gt_labels_subset: {gt_labels_subset.shape}")
            utils.plot_CM(predictions=tuned_predictions_subset,
                          gt_labels=gt_labels_subset,
                          tasks=target_class_dict.keys(),
                          model_name=model_args.model)
            
            
            acc = utils.compute_accuracy(tuned_predictions, gt_labels)
            print(f"[Validatoin] Accuracy using tuned thresholds: {acc:.4f}")
            
            youden = utils.comput_youden_idx(ground_truth=gt_labels, 
                                             preds=tuned_predictions, 
                                             tasks=tasks)
 

            f1 = utils.compute_f1_score(ground_truth=gt_labels, 
                                        preds=tuned_predictions, 
                                        tasks=tasks)

            # Average for eval_tasks
            avg_youden = sum(youden[task] for task in eval_tasks) / len(eval_tasks)
            avg_f1    = sum(f1[task]    for task in eval_tasks) / len(eval_tasks)

            results = {
                "accuracy": float(acc),
                "youden_index":    { task: float(youden[task]) for task in tasks },
                "f1_score":        { task: float(f1[task])    for task in tasks },
                "average_youden_index": float(avg_youden),
                "average_f1_score":      float(avg_f1)
            }

            save_dir = "results/confusion_matrices"
            json_path = os.path.join(save_dir, model_args.model, "evaluation_metrics.json")
            with open(json_path, "w") as jf:
                json.dump(results, jf, indent=4)

            print(f"Saved all validation metrics (including averages over eval_tasks) to {json_path}")


            filename = os.path.expanduser('~/repo/xai_thesis/third_party/results/' +  str(model_args.model) + '_tuned_' + str(model_args.metric) + '_thresholds.csv')
            with open(filename, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                # Header
                writer.writerow(['Task', 'Value'])
                # ROC AUC for each task
                for task, auc in optimal_thresholds.items():
                    writer.writerow([task, auc])

        
        # Save accuracy based on sigmoid threshold to csv file
        if model_args.save_acc_roc:
            filename = os.path.expanduser('~/repo/xai_thesis/third_party/results/'+  str(model_args.model) + '_sigmoid' +  str(model_args.sigmoid_threshold) + '.csv')
            with open(filename, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=' ')
                # Header
                writer.writerow(['Metric', 'Value'])
                # Accuracy
                writer.writerow(['Accuracy', acc])
                # ROC AUC for each task
                for task, auc in auroc.items():
                    writer.writerow([f'ROC AUC {task}', auc])
    
    print("********** Finished Eval mode **********")



def run_test_with_thresholds(model, model_args, tasks, test_loader, threshold_csv_path):
    """
    Runs the test set evaluation using previously saved optimal thresholds.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        model_args: Parsed command-line arguments.
        tasks (List[str]): List of task/class names.
        test_loader (DataLoader): DataLoader for the test dataset.
        threshold_csv_path (str): Path to CSV file with saved thresholds.

    Returns:
        dict: Dictionary with accuracy and AUROC per task.
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device for test:", DEVICE)
    model.to(DEVICE)
    model.eval()

    # Load thresholds
    thresholds_df = pd.read_csv(threshold_csv_path, header=0)
    thresholds = {row['Task']: float(row['Value']) for _, row in thresholds_df.iterrows()}

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(logits)

    predictions = utils.threshold_based_predictions(probs, thresholds, tasks)

    acc = utils.compute_accuracy(predictions, labels)
    print(f"[TEST] Accuracy using tuned thresholds: {acc:.4f}")

    youden = utils.comput_youden_idx(ground_truth=labels, preds=predictions, tasks=tasks)
    print("-" * 40)
    print(f"[TEST] Youden-index using tuned thresholds")
    for task in tasks:
        print(f"{task}: score = {youden[task]:.4f}")

    f1 = utils.compute_f1_score(ground_truth=labels, preds=predictions, tasks=tasks)
    print("-" * 40)
    print(f"[TEST] F1-score using tuned thresholds")
    for task in tasks:
        print(f"{task}: score = {f1[task]:.4f}")
    return {"accuracy": acc}


def main():
    model_args = parse_arguments()

    tasks = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
             'Lung Opacity', 'Lung Lesion', 'Edema',
             'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
             'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    # Create the model based on provided arguments
    model = get_model(model_args.model, tasks, model_args)
    print("Loaded model:", type(model))
    
    # Load checkpoint and prepare the model for inference
    model = load_checkpoint(model, model_args.ckpt)
    
    # Prepare the data loader for inference
    data_loader = prepare_data(model_args=model_args)
    
    # Run inference on one batch and print the predictions
    logits = model_run(model=model, data_loader=data_loader)

    eval_model(model_args=model_args, data_loader=data_loader, tasks=tasks, logits=logits)

    if model_args.run_test:
        #debug_densenet_threshold_path = '/home/fkirchhofer/repo/xai_thesis/third_party/results/DenseNet121_tuned_f1_thresholds.csv'
        debug_resnet_threshold_path = '/home/fkirchhofer/repo/xai_thesis/third_party/results/ResNet152_tuned_f1_thresholds.csv'
        #debug_inception_threshold_path = ''


        test_loader = prepare_data(model_args)  # or a specific `prepare_test_data()`
        run_test_with_thresholds(model, model_args, tasks, test_loader, debug_resnet_threshold_path)


if __name__ == '__main__':
    """
    When run directly: The condition evaluates to True, 
                    and the main() function is executed.
    When imported: The condition is False, 
                    and the main() function is not executed automatically. 
                    This is especially useful if you want to import functions
                    or classes from your script into another module without 
                    running the whole pipeline.
    """
    main()

