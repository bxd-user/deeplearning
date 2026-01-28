import os
import math
import zipfile
import urllib.request
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =====================================================
# 配置
# =====================================================
DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

IMG_SIZE = 64
PATCH_SIZE = 8
NUM_CLASSES = 200

D_MODEL = 384
N_HEADS = 6
N_LAYERS = 6
MLP_RATIO = 4

BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 自动下载 & 解压
# =====================================================
def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("[INFO] Downloading Tiny-ImageNet...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("[INFO] Download complete")

    if not os.path.exists(DATASET_DIR):
        print("[INFO] Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        print("[INFO] Extraction complete")

    print("[INFO] Dataset ready")

# =====================================================
# Dataset
# =====================================================
class TinyImageNet(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.samples = []
        self.transform = transform

        if split == "train":
            classes = sorted(os.listdir(root))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

            for cls in classes:
                img_dir = os.path.join(root, cls, "images")
                for img in os.listdir(img_dir):
                    self.samples.append(
                        (os.path.join(img_dir, img), self.class_to_idx[cls])
                    )

        elif split == "val":
            ann_file = os.path.join(root, "val_annotations.txt")
            img_dir = os.path.join(root, "images")

            classes = sorted(
                set(line.split()[1] for line in open(ann_file))
            )
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

            with open(ann_file) as f:
                for line in f:
                    img, cls = line.split()[:2]
                    self.samples.append(
                        (os.path.join(img_dir, img), self.class_to_idx[cls])
                    )

        print(f"[INFO] Loaded {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# =====================================================
# ViT 模型
# =====================================================
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(
            3, D_MODEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * MLP_RATIO),
            nn.GELU(),
            nn.Linear(D_MODEL * MLP_RATIO, D_MODEL),
        )
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, D_MODEL)
        )

        self.blocks = nn.Sequential(
            *[EncoderBlock() for _ in range(N_LAYERS)]
        )
        self.head = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        return self.head(x[:, 0])

# =====================================================
# Train / Eval
# =====================================================
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            loss_sum += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total

def train():
    prepare_dataset()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_set = TinyImageNet(
        os.path.join(DATASET_DIR, "train"),
        split="train",
        transform=tf,
    )
    val_set = TinyImageNet(
        os.path.join(DATASET_DIR, "val"),
        split="val",
        transform=tf,
    )

    train_loader = DataLoader(
        train_set, BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = ViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()

        val_loss, val_acc = evaluate(model, val_loader)
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

if __name__ == "__main__":
    train()
