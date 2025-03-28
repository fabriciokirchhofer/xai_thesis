import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode



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
        image = read_image(path=img_path, mode=ImageReadMode.RGB).float()

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


def get_dataloader(annotations_file, img_dir, transform, batch_size=64, shuffle=False):
    """
    Factory function to create a DataLoader for the CustomImageDataset.
    Args:
        annotations_file (str): path to csv file with labels / annotations
        img_dir (str): pyth to main image directory
        transform (callable, optional): transformations by default False 
        batch_size (int): by default 64
        shuffle (bool): by default False

    Returns:
        DataLoader: Iterable instance of BasicImageDataset
    """
    if batch_size == 64:
        num_workers = 16
    elif batch_size == 16:
        num_workers = 4
    else:
        num_workers = 1

    dataset = BasicImageDataset(annotations_file, img_dir, transform=transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def compute_dataset_statistics(data_loader):
    """
    Computes and returns the mean and standard deviation of the dataset.
    Args:
        data_loader (callable) iterable dataset containing images along axis 0

    returns: A torch.tensor for mean and one for std 
    """
    total_mean = 0.0
    total_std = 0.0
    num_samples = 0

    print("Starts calculating mean and std from dataset.")
    for images, _ in data_loader:
        batch_size = images.size(0)
        # Flatten height and width dimensions
        images = images.view(batch_size, images.size(1), -1)
        total_mean += images.mean(2).sum(0)
        total_std += images.std(2).sum(0)
        num_samples += batch_size

    mean = total_mean / num_samples
    std = total_std / num_samples
    return mean, std

if __name__ == '__main__':
    # Example usage for testing purposes
    val_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/valid.csv'
    val_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0/'

    test_data_labels_path = '/home/fkirchhofer/data/CheXpert-v1.0/test.csv'
    test_data_img_path = '/home/fkirchhofer/data/CheXpert-v1.0'
    
    # Optionally define a basic transform, e.g., resizing the image
    from torchvision import transforms
    basic_transform = transforms.Compose([
        transforms.Resize((320, 320))
    ])
    
    # Create a DataLoader using the factory function
    loader = get_dataloader(val_data_labels_path, val_data_img_path, basic_transform, batch_size=16)
    
    # Test image size retrieval from the dataset
    dataset_instance = BasicImageDataset(val_data_labels_path, val_data_img_path)
    print("Image size for first image before transform:", dataset_instance.get_img_size(0))
    
    # Optionally compute and print dataset statistics (mean and std)
    # mean, std = compute_dataset_statistics(loader)
    # print("Dataset mean:", mean)
    # print("Dataset std:", std)
