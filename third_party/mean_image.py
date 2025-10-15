import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from run_models import prepare_data, parse_arguments
import torchvision.utils as vutils

args = parse_arguments()

data_loader = prepare_data(args)

sum_image = None
count = 0

for batch in tqdm(data_loader, desc="Computing mean image"):
    inputs = batch[0]  # assuming batch = (inputs, labels)
    # If batch_size > 1: shape = [B, C, H, W]
    for img in inputs:
        if sum_image is None:
            sum_image = img.clone()
        else:
            sum_image += img
        count += 1

mean_image = sum_image / count
print(f"Computed mean image from {count} validation images.")

torch.save(mean_image, "mean_image.pt")
vutils.save_image(mean_image, "mean_image.png")

print(f"Shape of mean_image: {mean_image.squeeze().squeeze().detach().numpy().shape}")
print(f"The mean image: {mean_image}")

