import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

# import pretrainedmodels



class BasicImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, test=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.test = test
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(path=img_path, mode=ImageReadMode.RGB).float()/255
        if self.test:
            labels = self.img_labels.iloc[idx, 1:].values.astype(float)
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = self.img_labels.iloc[idx, 5:].values.astype(float)
            labels = torch.tensor(labels, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels
    
    def get_img_size(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        return image.size()


def get_dataloader(annotations_file, img_dir, transform=None, batch_size=1, shuffle=False, test=False):
    """
    Factory function to create a DataLoader for the CustomImageDataset.
    Args:
        annotations_file (str): path to csv file with labels / annotations
        img_dir (str): pyth to main image directory
        transform (callable, optional): transformations by default False 
        batch_size (int): by default 1
        shuffle (bool): by default False
        test (bool): In test mode the csv file with the lables with skip the first 5 columns. By default False

    Returns:
        DataLoader: Iterable instance of BasicImageDataset
    """
    if batch_size == 64:
        num_workers = 16
    elif batch_size == 16:
        num_workers = 4
    else:
        num_workers = 0
    print(f"Batch_size: {batch_size} with num_workers: {num_workers}")

    dataset = BasicImageDataset(annotations_file, img_dir, transform=transform, test=test)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def compute_dataset_statistics(data_loader, device='cpu'):
    """
    Computes per‑channel mean and std over a dataset using PyTorch.

    Args:
        data_loader: yields (images, labels), where images is a torch.Tensor
                     of shape (B, C, H, W), dtype=torch.float32.
        device:      device where tensors should live during accumulation.

    Returns:
        mean: torch.Tensor of shape (C,)
        std:  torch.Tensor of shape (C,)
    """
    # Grab first batch to get channel count
    first_batch = next(iter(data_loader))[0]
    C = first_batch.size(1)

    # Accumulators
    cnt = 0
    sum_channels = torch.zeros(C, device=device)
    sum_sq_channels = torch.zeros(C, device=device)
    print("Start calculating statistics with chatGPT version torch")

    with torch.no_grad():
        for images, _ in data_loader:
            # Move to desired device
            images = images.to(device)       # (B, C, H, W)
            B, C, H, W = images.shape

            # Flatten height & width: shape → (B, C, H*W)
            imgs = images.view(B, C, -1)

            # Sum over batch & pixels
            sum_channels    += imgs.sum(dim=[0, 2])
            sum_sq_channels += (imgs ** 2).sum(dim=[0, 2])
            cnt += B * H * W

    # Mean & std
    mean = sum_channels    / cnt
    var  = sum_sq_channels / cnt - mean**2
    std  = torch.sqrt(var)

    return mean, std


def compute_dataset_statistics_no_resize(dataset):
    """
    Computes overall mean and standard deviation for a dataset without resizing images.
    Args:
        dataset (torch.utils.data.Dataset): Dataset returning (image, label) pairs.
    Returns:
        mean, std (torch.tensors) computed over all pixels.
    """
    total_sum = torch.zeros(3)      # assuming 3 channels (RGB)
    total_sq_sum = torch.zeros(3)
    total_pixels = 0

    print("Starts calculating statistics without resizing")

    for i in range(len(dataset)):
        image, _ = dataset[i]  # image shape: [C, H, W] with potentially different H, W
        # Flatten each image to shape [C, H*W]
        image = image.view(image.size(0), -1)
        total_sum += image.sum(dim=1)
        total_sq_sum += (image ** 2).sum(dim=1)
        total_pixels += image.size(1)

    mean = total_sum / total_pixels
    std = torch.sqrt(total_sq_sum / total_pixels - mean ** 2)
    return mean, std



if __name__ == '__main__':
    # To calculate stats - Do not specify train/val/test in the img_path as it will take it from the labels_path.
    train_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/train.csv'
    train_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'


    val_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/valid.csv'
    val_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'

    test_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'
    test_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'
    
    # Optionally define a basic transform, e.g., resizing the image
    from torchvision import transforms
    basic_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((320, 320))
    ])

    
    #Create a DataLoader using the factory function
    loader = get_dataloader(train_data_labels_path, train_data_img_path, transform=basic_transform, batch_size=64)
    
    # Test image size retrieval from the dataset
    # dataset_instance = BasicImageDataset(val_data_labels_path, val_data_img_path)
    # print("Image size for first image before transform:", dataset_instance.get_img_size(0))
    
    # Optionally compute and print dataset statistics (mean and std)
    mean, std = compute_dataset_statistics(loader)
    print("Dataset mean:", mean)
    print("Dataset std:", std)

