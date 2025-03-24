import os
import models
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
#from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode



#******************** Utils ********************

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(path=img_path, mode=ImageReadMode.RGB).float()

        labels = self.img_labels.iloc[idx, 1:].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels
    
    def __size__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        return image.size()



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


#******************** Model call ********************

tasks = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
    'Fracture', 'Support Devices', 'No Finding'
]

parser = argparse.ArgumentParser(description = 'Model Playground')
parser.add_argument('--model_uncertainty',type=bool, default=True, help='Model uncertainty')
parser.add_argument('--pretrained',type=bool, default=True, help='Use pre-trained model')
parser.add_argument('--model', type=str, default='DenseNet121', help='specify model name')
model_args=parser.parse_args()

model = models.DenseNet121(tasks=tasks, model_args=model_args)
print(type(model))

ckpt_d_3class_1 = torch.load('pretrainedmodels/densenet121/uncertainty/densenet_3class_1/epoch=2-chexpert_competition_AUROC=0.88_v2.ckpt') # torch.Size([42, 1024])
ckpt_d_ignore_1 = torch.load('pretrainedmodels/densenet121/uncertainty/densenet_ignore_1/epoch=2-chexpert_competition_AUROC=0.87_v1.ckpt') # torch.Size([14, 1024])
ckpt_d_ones_1 = torch.load('pretrainedmodels/densenet121/uncertainty/densenet_ones_1/epoch=2-chexpert_competition_AUROC=0.88_v0.ckpt') # torch.Size([14, 1024])
ckpt_d_zeros_3 = torch.load('pretrainedmodels/densenet121/uncertainty/densenet_zeros_3/epoch=2-chexpert_competition_AUROC=0.86_v1.ckpt') # torch.Size([14, 1024])
#print(f"Checkpoint type: {type(ckpt_d_3class_1)}")

# if 'state_dict' in checkpoint:
#     print(f"state_dict in checkpoint")
# else:
#     print("state_dict not in checkpoint")

# state_dict will be a dict subclass that remembers the order entries were added
state_dict = ckpt_d_3class_1.get('state_dict', ckpt_d_3class_1)
#print(f"state_dict type: {type(state_dict)}\n\n")
#print(f"state_dict keys: {list(state_dict.keys())}")
state_dict = remove_prefix(state_dict, "model.")

model.load_state_dict(state_dict=state_dict)
model.eval()


#******************** Set up dataset ********************

# Get paths and define initial transfrom needed for processing
test_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'
test_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0'
basic_transform = transforms.Compose([
    transforms.Resize((320, 320))
])
batch_size = 1

# Create dataset
dataset = CustomImageDataset(
    annotations_file=test_data_labels_path,
    img_dir=test_data_img_path,
    transform=basic_transform
)

print("****** Loading data ******")
print(f"size of dataset {dataset.__size__(0)}")
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) # Wraps an iterable around the dataset for easier access

"""
print("****** Calculating meand & std ******")
mean = 0.0
std = 0.0
n_samples = 0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1) # [batch, channels, H*W]

    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_samples

mean /= n_samples
std /= n_samples

print(f"Dataset mean: {mean}")
print(f"Dataset std: {std}")
"""


#******************** Pass image ********************

# Open image 
# img_path = '/home/fkirchhofer/data/CheXpert-v1.0/test/patient64741/study1/view1_frontal.jpg'
# img = Image.open(img_path).convert('RGB')

mean = torch.tensor([128.0847, 128.0847, 128.0847])
std = torch.tensor([74.5220, 74.5220, 74.5220])

# Define and apply preprocessing transformations
inference_transform  = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

print("****** Recreating dataset ******")
dataset = CustomImageDataset(
    annotations_file=test_data_labels_path,
    img_dir=test_data_img_path,
    transform=inference_transform  # now includes normalization
)

dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) # Wraps an iterable around the dataset for easier access

# Turn off gradients for inference
with torch.no_grad():
    for images, labels in dataset:
        # If available, move model and input to GPU
        if torch.cuda.is_available():
            print("GPU available")
            images = images.cuda()
            labels = labels.cuda()
            model = model.cuda()
        else:
            print("Run without GPU")       
        logits = model(images)

        probs = F.softmax(logits.view(-1, 14, 3), dim=2)
        print(f"Probs:\n {probs}")
        break

# Print the probabilities for each task (label)
# for task, prob in zip(tasks, probs):
#     print(f"{task}: {prob.view(16,3,14)():.4f}")

# Assuming: 
# probs: (16, 42) tensor
# tasks: list of 14 task names (length = 14)
# class_names: ['uncertain', 'certain', 'ignore']

class_names = ['neg-zeros', 'uncertain', 'pos-ones']

for i, sample_probs in enumerate(probs):  # loop over samples
    print(f"Sample {i+1}:")
    
    for task_name, task_probs in zip(tasks, sample_probs):
        #task_probs = torch.nn.functional.softmax(task_probs, dim=0)  # if not already softmaxed
        prob_str = ", ".join(
            f"{cls}: {prob.item():.4f}" for cls, prob in zip(class_names, task_probs)
        )
        print(f"  {task_name}: {prob_str}")
    
    print("-" * 40)