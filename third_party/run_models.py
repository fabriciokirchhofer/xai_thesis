import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from third_party import models
from third_party import utils
from third_party import dataset
import csv
import pandas as pd


# Calculated average AUROC over atelectasis, cardiomegaly, consolidation, edema, and pleural effusion = 0.8341 (val)
ckpt_d_ignore_1 = 'pretrainedmodels/densenet121/uncertainty/densenet_ignore_1/epoch=2-chexpert_competition_AUROC=0.87_v1.ckpt' # torch.Size([14, 1024])

debug_path_to_ckpt_d_ignore_1 = '/home/fkirchhofer/repo/xai_thesis/third_party/pretrainedmodels/densenet121/uncertainty/densenet_ignore_1/epoch=2-chexpert_competition_AUROC=0.87_v1.ckpt'
debug_path_to_ckpt_r_ignore_2 = '/home/fkirchhofer/repo/xai_thesis/third_party/pretrainedmodels/resnet152/resnet_ignore_2/epoch=2-chexpert_competition_AUROC=0.86.ckpt' # AUROC during val = 0.7969021549350594
debut_path_to_ckpt_i_irgnore_2 = '/home/fkirchhofer/repo/xai_thesis/third_party/pretrainedmodels/inceptionv4/inception_ignore_2/epoch=2-chexpert_competition_AUROC=0.86_v2.ckpt'

# Parse arguments -> Argumente Zerlegung
def parse_arguments():
    parser = argparse.ArgumentParser(description="Models exploration")
    parser.add_argument('--pretrained',type=bool, default=True, help='Use pre-trained model')
    parser.add_argument('--model_uncertainty', type=bool, default=False, help='Use model uncertainty') # Inf not further used it can be removed
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size which will be passed to the model')
    parser.add_argument('--model', type=str, default='DenseNet121', help='specify model name')
    parser.add_argument('--ckpt', type=str, default=debug_path_to_ckpt_d_ignore_1, help='Path to checkpoint file')

    parser.add_argument('--save_acc_roc', type=bool, default=False, help='Save accuracy and auroc during validation to csv file')
    parser.add_argument('--sigmoid_threshold', type=float, default=0.5, help='The threshold to activate sigmoid function. Used for model evaluation in validation.')
    parser.add_argument('--tune_thresholds', type=bool, default=False, help='If True, find optimal per-class thresholds using F1 score. Will save it.')
    parser.add_argument('--metric', type=str, default='f1', help='Choose evaluation evaluation metric. Can be "f1" or "youden".')
    parser.add_argument('--plot_roc', type=bool, default=False, help='Plot the ROC curves for each task. Default false.')
    parser.add_argument('--run_test', type=bool, default=False, help='Runs the test set for evaluation. Needs thresholds from tune_thresholds as a csv file.')
    return parser.parse_args()

def get_model(model, tasks, model_args):
    # Mapping to choose the right model class
    model_map = {
        'DenseNet121': models.DenseNet121,
        'ResNet152': models.ResNet152,
        'Inceptionv4': models.Inceptionv4
    }

    # Return value of dict model_map {key: value}
    # models.DenseNet121 will only be used if no valid or recognized value is passed in the model_args
    model_class = model_map.get(model, models.DenseNet121)
    # print(f"Model_class {model_class}")
    # print(f"Model_class type {type(model_class)}")
    return model_class(tasks=tasks, model_args=model_args)


# Load checkpoint and its parameters
def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = utils.remove_prefix(state_dict, "model.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Prep dataset
def prepare_data(model_args):
    
    if not model_args.run_test:
        print("Prepare validation data...") 

        # Data paths
        val_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/valid.csv'
        val_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'

        # Hardcoded normalization parameters (could also be computed but takes some time to run)
        # Stats calculated with val set
        # val_mean = torch.tensor([0.5041, 0.5041, 0.5041])
        # val_std = torch.tensor([0.2915, 0.2915, 0.2915])

        # Stats calculated with train set
        val_mean = torch.tensor([0.5031, 0.5031, 0.5031])
        val_std = torch. tensor([0.2914, 0.2914, 0.2914])
        
        # Define inference transformation pipeline
        inference_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((320, 320)),
            transforms.Normalize(mean=val_mean.tolist(), std=val_std.tolist())
                       
        ])
        
        data_loader = dataset.get_dataloader(
            annotations_file=val_data_labels_path,
            img_dir=val_data_img_path,
            transform=inference_transform,
            batch_size=model_args.batch_size,
            shuffle=False,
            test=False
        )


    if model_args.run_test:
        print("Prepare test data...")
        test_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'
        test_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0'

        # Hardcoded normalization parameters (could also be computed but takes some time to run)
        test_mean = torch.tensor([128.0847, 128.0847, 128.0847])/255
        test_std = torch.tensor([74.5220, 74.5220, 74.5220])/255

        # Define inference transformation pipeline
        tetst_inference_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((320, 320)),
            transforms.Normalize(mean=test_mean.tolist(), std=test_std.tolist())
        ])
    
        data_loader = dataset.get_dataloader(
            annotations_file=test_data_labels_path,
            img_dir=test_data_img_path,
            transform=tetst_inference_transform,
            batch_size=model_args.batch_size,
            shuffle=False,
            test=True
        )

    return data_loader


# Run the model
def model_run(model, data_loader, tasks, model_args):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device for evaluation:", device)
    model.to(device)
  
    if not model_args.model_uncertainty: # If 3class model is used
        print("Running in standard multi-label (sigmoid) mode.\n")
        all_logits = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())

        # Concatenate batch results from list into one torch tensor
        all_logits = torch.cat(all_logits, dim=0)



    # # Old - this 3class model evaluation must be updated in order to work properly
    # else: # if model_uncertainty is True i.e. 3class model is used.
    #     class_names = ['neg-zeros', 'uncertain', 'pos-ones']  
    #     for images, labels in data_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         logits = model(images)
    #         # Dynamically compute the number of tasks from the tasks list
    #         probs = F.softmax(logits.view(-1, len(tasks), 3), dim=1)
    #         print("Probabilities:\n", probs)
    #         # Print results for each sample
    #         for i, sample_probs in enumerate(probs):
    #             print(f"Sample {i+1}:")
    #             for task_name, task_probs in zip(tasks, sample_probs):
    #                 prob_str = ", ".join(f"{cls}: {prob.item():.4f}" for cls, prob in zip(class_names, task_probs))
    #                 print(f"  {task_name}: {prob_str}")
    #             print("-" * 40)
    #         break  # Remove break to run on the whole dataset
  
    return all_logits


def eval_model(model_args, data_loader, tasks, logits):
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

    print("****Start evaluation mode****")
    # Concatenate batch results from list into one torch tensor
    # Calculate probabilities obtained from the logits
    probs = torch.sigmoid(logits)

    #******************** get max prob per view start ********************
    df = utils.extract_study_id(mode=model_args.run_test)

    # Add study_id col to df
    #print("Current df:", df.head())

    # Convert probs and gt_labels to df
    prob_df = pd.DataFrame(probs.detach().cpu().numpy(), columns=tasks)
    #print("Current df:", prob_df.head())
    gt_df   = pd.DataFrame(gt_labels.detach().cpu().numpy(), columns=tasks)
    print("******************Shape of prob_df:", prob_df.shape)

    # Match each row of predictions and gt to the df from the csv file
    prob_df['study_id'] = df['study_id']
    gt_df['study_id'] = df['study_id']

    # Group by study_id and take only the maximum predicted probability per study.
    agg_prob = prob_df.groupby('study_id').max()
    agg_gt = gt_df.groupby('study_id').max()
    print("******************Shape of agg_prob:", agg_prob.shape)
 
    probs = torch.tensor(agg_prob.values)
    gt_labels   = torch.tensor(agg_gt.values)
    #******************** get max prob per view end ********************

    if not model_args.run_test:
        predictions = utils.threshold_based_predictions(probs, model_args.sigmoid_threshold, tasks)

        # Move to CPU (logits from model_run are already concatenated across batches)
        probs = probs.cpu()
        predictions = predictions.cpu()


        acc = utils.compute_accuracy(predictions, gt_labels)
        print("-" * 40)
        print(f"Overall Accuracy with default threshold {model_args.sigmoid_threshold}: {acc:.4f}")

        auroc = utils.auroc(predictions=probs, ground_truth=gt_labels, tasks=tasks)
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

        # If threshold tuning is enabled, compute optimal per-class thresholds based on F1 score
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


            filename = ('results/' +  str(model_args.model) + '_tuned_' + str(model_args.metric) + '_thresholds.csv')
            with open(filename, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                # Header
                writer.writerow(['Task', 'Value'])
                # ROC AUC for each task
                for task, auc in optimal_thresholds.items():
                    writer.writerow([task, auc])

        
        # Save accuracy based on sigmoid threshold to csv file
        if model_args.save_acc_roc:
            filename = ('results/'+  str(model_args.model) + '_sigmoid' +  str(model_args.sigmoid_threshold) + '.csv')
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
    return 0



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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device for test:", device)
    model.to(device)
    model.eval()

    # Load thresholds
    thresholds_df = pd.read_csv(threshold_csv_path, header=0)
    thresholds = {row['Task']: float(row['Value']) for _, row in thresholds_df.iterrows()}

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
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
    
    """
    # TODO: Based on Mikes recommendation: Check two ways how to replace and handle parse_arguments
    import json
    with open('path.json', encoding='utf-8') as f:
        config = json.load(f)

    from itertools import permutations
    """
    tasks = [
        'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion' , 'Edema' ,
        'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    
    # Create the model based on provided arguments
    model = get_model(model_args.model, tasks, model_args)
    print("Loaded model:", type(model))
    
    # Load checkpoint and prepare the model for inference
    model = load_checkpoint(model, model_args.ckpt)
    
    # Prepare the data loader for inference
    data_loader = prepare_data(model_args=model_args)
    
    # Run inference on one batch and print the predictions
    logits = model_run(model=model, data_loader=data_loader, tasks=tasks, model_args=model_args)

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

