import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
import models
import utils
import dataset
import csv


ckpt_d_3class_1 = 'pretrainedmodels/densenet121/uncertainty/densenet_3class_1/epoch=2-chexpert_competition_AUROC=0.88_v2.ckpt' # torch.Size([42, 1024])
ckpt_d_ignore_1 = 'pretrainedmodels/densenet121/uncertainty/densenet_ignore_1/epoch=2-chexpert_competition_AUROC=0.87_v1.ckpt' # torch.Size([14, 1024])
ckpt_d_ones_1 = 'pretrainedmodels/densenet121/uncertainty/densenet_ones_1/epoch=2-chexpert_competition_AUROC=0.88_v0.ckpt' # torch.Size([14, 1024])
ckpt_d_zeros_3 = 'pretrainedmodels/densenet121/uncertainty/densenet_zeros_3/epoch=2-chexpert_competition_AUROC=0.86_v1.ckpt' # torch.Size([14, 1024])

debug_path_to_ckpt_d_ignore_1 = '/home/fkirchhofer/repo/xai_thesis/third_party/pretrainedmodels/densenet121/uncertainty/densenet_ignore_1/epoch=2-chexpert_competition_AUROC=0.87_v1.ckpt'



# Parse arguments -> Argumente Zerlegung
def parse_arguments():
    parser = argparse.ArgumentParser(description="Models exploration")
    parser.add_argument('--model_uncertainty', type=bool, default=False, help='Use model uncertainty')
    parser.add_argument('--pretrained',type=bool, default=True, help='Use pre-trained model')
    parser.add_argument('--model', type=str, default='DenseNet121', help='specify model name')
    parser.add_argument('--ckpt', type=str, default=debug_path_to_ckpt_d_ignore_1, help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size which will be passed to the model')
    parser.add_argument('--save', type=bool, default=False, help='Save accuracy and auroc to csv file')
    parser.add_argument('--sigmoid_threshold', type=float, default=0.5, help='The threshold to activate sigmoid function. Used for model evaluation.')
    return parser.parse_args()

def get_model(model_name, tasks, model_args):
    # Mapping to choose the right model class
    model_map = {
        'DenseNet121': models.DenseNet121,
        'ResNet152': models.ResNet152,
        'Inceptionv4': models.Inceptionv4
    }

    # Return value of dict model_map {key: value}
    # models.DenseNet121 will only be used if no valid or recognized value is passed in the model_args
    model_class = model_map.get(model_name, models.DenseNet121)
    # print(f"Model_class {model_class}")
    # print(f"Model_class type {type(model_class)}")
    return model_class(tasks=tasks, model_args=model_args)


# Load checkpoint
def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = utils.remove_prefix(state_dict, "model.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Prep dataset
def prepare_data(model_args):
    # Data paths
    val_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/valid.csv'
    val_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'

    # test_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'
    # test_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0'
    
    # Hardcoded normalization parameters (could also be computed but takes some time to run)
    val_mean = torch.tensor([128.2917, 128.2917, 128.2917])
    val_std = torch.tensor([74.3154, 74.3154, 74.3154])

    # test_mean = torch.tensor([128.0847, 128.0847, 128.0847])
    # test_std = torch.tensor([74.5220, 74.5220, 74.5220])
    
    # Define inference transformation pipeline
    inference_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=val_mean.tolist(), std=val_std.tolist())
    ])
    
    data_loader = dataset.get_dataloader(
        annotations_file=val_data_labels_path,
        img_dir=val_data_img_path,
        transform=inference_transform,
        batch_size=model_args.batch_size,
        shuffle=False
    )

    # For test case
    # data_loader = dataset.get_dataloader(
    #     annotations_file=test_data_labels_path,
    #     img_dir=test_data_img_path,
    #     transform=inference_transform,
    #     batch_size=model_args.batch_size,
    #     shuffle=False
    # )
    return data_loader



def compute_accuracy(predictions, labels):
    """
    Computes overall binary accuracy across all tasks.
    """
    correct = (predictions == labels).float().sum()
    total = labels.numel()
    accuracy = correct / total
    return accuracy.item()


# Run the model
def model_run(model, data_loader, tasks, model_args):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    
    if not model_args.model_uncertainty: # If 3class model is used
        print("Running in standard multi-label (sigmoid) mode.")
        all_predictions = []
        all_labels = []
        # Compute logits
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                #print("logits dim:", logits)
                # Dynamically compute the number of tasks from the tasks list
                probs = torch.sigmoid(logits)
                predictions = (probs >= model_args.sigmoid_threshold).float()
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # Concatenate batch results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = compute_accuracy(all_predictions, all_labels)
        print(f"Overall Accuracy: {acc:.4f}")
        auroc = utils.auroc(predictions=all_predictions, ground_truth=all_labels, tasks=tasks)
        print(f"AUROC: {auroc}")

        # # Print probabilities
        # for i, sample_probs in enumerate(probs):
        #     print(f"Sample {i+1}:")
        #     for task_name, task_probs in zip(tasks, sample_probs):
        #         print(f"  {task_name}: {task_probs:.4f}")
        #     print("-" * 40)
        #     break  # Remove break if you want to run on the whole dataset

    else: # if model_uncertainty is True i.e. 3class model is used.
        class_names = ['neg-zeros', 'uncertain', 'pos-ones']  
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            # Dynamically compute the number of tasks from the tasks list
            probs = F.softmax(logits.view(-1, len(tasks), 3), dim=1)
            print("Probabilities:\n", probs)
            # Print results for each sample
            for i, sample_probs in enumerate(probs):
                print(f"Sample {i+1}:")
                for task_name, task_probs in zip(tasks, sample_probs):
                    prob_str = ", ".join(
                        f"{cls}: {prob.item():.4f}" for cls, prob in zip(class_names, task_probs)
                    )
                    print(f"  {task_name}: {prob_str}")
                print("-" * 40)
            break  # Remove break if you want to run on the whole dataset
    
    
    if model_args.save:
        filename = ('results/sigmoid' +  str(model_args.sigmoid_threshold) + '.csv')
        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=' ')
            # Header
            writer.writerow(['Metric', 'Value'])
            # Accuracy
            writer.writerow(['Accuracy', acc])
            # ROC AUC for each task
            for task, auc in auroc.items():
                writer.writerow([f'ROC AUC {task}', auc])


def main():
    model_args = parse_arguments()
    
    """
    # TODO: Check two ways how to replace and handle parse_arguments
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

    # Old output - delete later when results seem reasonable
    # tasks = [
    #     'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #     'Enlarged Cardiomediastinum', 'Lung Lesion', 'Lung Opacity',
    #     'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
    #     'Fracture', 'Support Devices', 'No Finding'
    # ]
    
    # Create the model based on provided arguments
    model = get_model(model_args.model, tasks, model_args)
    print("Loaded model:", type(model))
    
    # Load checkpoint and prepare the model for inference
    model = load_checkpoint(model, model_args.ckpt)
    
    # Prepare the data loader for inference
    data_loader = prepare_data(model_args=model_args)
    
    # Run inference on one batch and print the predictions
    model_run(model=model, data_loader=data_loader, tasks=tasks, model_args=model_args)

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

