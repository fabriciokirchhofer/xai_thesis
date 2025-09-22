import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from run_models import prepare_data, parse_arguments
import torchvision.utils as vutils

args = parse_arguments()

# # 🔁 Define your transform (match your model’s training transform)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(),  # CheXpert images are single-channel
#     transforms.ToTensor(),   # Converts to shape [1, H, W], values in [0, 1]
# ])

# # 📦 Dummy Dataset class (replace with your real one)
# class CheXpertValDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = sorted([
#             os.path.join(root_dir, f) for f in os.listdir(root_dir)
#             if f.endswith('.jpg') or f.endswith('.png')
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
#         if self.transform:
#             img = self.transform(img)
#         return img

# # 🔍 Define dataset and loader
# val_dataset = CheXpertValDataset(root_dir='path/to/chexpert/valid', transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

data_loader = prepare_data(args)

# 🔢 Accumulate mean image
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

# 💾 Optional: Save mean image for future use
torch.save(mean_image, "mean_image.pt")
vutils.save_image(mean_image, "mean_image.png")

print(f"Shape of mean_image: {mean_image.squeeze().squeeze().detach().numpy().shape}")
print(f"The mean image: {mean_image}")

