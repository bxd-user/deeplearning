import os
import urllib.request
import zipfile
import shutil
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 32
num_epochs = 5
learning_rate = 0.01
num_classes = 200

def download_tiny_imagenet(root="data"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    extract_path = os.path.join(root, "tiny-imagenet-200")

    os.makedirs(root, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading Tiny-ImageNet...")
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(extract_path):
        print("Extracting Tiny-ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)

    return extract_path

def prepare_val_folder(val_dir):
    anno_file = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")

    with open(anno_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        img_name, class_id = line.split("\t")[:2]
        class_dir = os.path.join(val_dir, class_id)

        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(images_dir, img_name)
        dst = os.path.join(class_dir, img_name)

        if os.path.exists(src):
            shutil.move(src, dst)

    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    print("Validation folder prepared.")

data_root = download_tiny_imagenet()

train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")

if os.path.exists(os.path.join(val_dir, "images")):
    prepare_val_folder(val_dir)

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
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),

    nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(4096, num_classes)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100. * correct / total

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100. * correct / total

for epoch in range(num_epochs):
    start = time.time()

    train_loss, train_acc = train_one_epoch(net, train_loader)
    val_loss, val_acc = validate(net, val_loader)

    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
        f"Time: {time.time() - start:.1f}s"
    )

    if (epoch + 1) % 5 == 0:
        torch.save(net.state_dict(), f"alexnet_tiny_epoch_{epoch+1}.pth")

print("Training finished!")
