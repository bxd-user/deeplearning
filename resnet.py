import os
import time

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

data_root = "data/tiny-imagenet-200"
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=192,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=192,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net=nn.Sequential(
    
)