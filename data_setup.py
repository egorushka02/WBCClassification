"""
    Contains functionality for creating PyTorch DataLoaders for
    image classification data
"""


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

NUM_WORKERS = os.cpu_count()

class WBCDataset(Dataset):
  """
  Defining a custom dataset class for the WBC (White Blood Cells) dataset
  """
  
  def __init__(self, root_dir, transform = None):
    self.root_dir = root_dir
    self.classes = os.listdir(root_dir)
    self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
    self.images = self._get_images()
    self.transform = transform

  def _get_images(self):
    images = []
    for cls in self.classes:
      class_path = os.path.join(self.root_dir, cls)
      for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        images.append((img_path, self.class_to_idx[cls]))
    return images

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_path, label = self.images[idx]
    image = Image.open(img_path).convert('RGB')

    if self.transform:
      image = self.transform(image)
    return image, label


def create_dataloaders(
        data_path:str,
        bs_train:int,
        bs_val:int,
        transforms: transforms.Compose
):
    """Creates training and testing DataLoaders.

  Takes a path to data, train and validation batch sizes and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    data_path: path to your images.
    bs_train: batch_size for training dataloader
    bs_val: batch size for validation data loader
    transform: torchvision transforms to perform on training and validation data.

  Returns:
    A tuple of (train_dataloader, val_dataloader).
  """
    wbc_dataset = WBCDataset(data_path, transform=transforms)

    train_size = int(0.8*len(wbc_dataset))
    val_size = len(wbc_dataset) - train_size
    train_pbc_data, val_pbc_data = torch.utils.data.random_split(wbc_dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset = train_pbc_data,
                                  batch_size=bs_train,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_pbc_data,
                                batch_size=bs_val,
                                num_workers=NUM_WORKERS,
                                shuffle=False)
    
    return train_dataloader, val_dataloader