import os
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_tiny_imagenet(zip_path, root="data"):
    extract_path = os.path.join(root, "tiny-imagenet-200")
    if not os.path.exists(extract_path):
        print("[INFO] Extracting Tiny-ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
    return extract_path

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = self.proj(x)          # (B, C, H', W')
        x = x.flatten(2)          # (B, C, N)
        x = x.transpose(1, 2)     # (B, N, C)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = [
            t.view(B, N, self.num_heads, self.head_dim)
             .transpose(1, 2)
            for t in qkv
        ]

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8,
                 embed_dim=384, depth=6, num_heads=6, num_classes=200):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.blocks = nn.Sequential(
            *[EncoderBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = prepare_tiny_imagenet(
        "data/tiny-imagenet-200.zip"
    )

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    train_set = datasets.ImageFolder(
        os.path.join(data_root, "train"),
        transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=128,
        shuffle=True, num_workers=4
    )

    model = VisionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        total = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch}: acc={correct/total:.3f}")

if __name__ == "__main__":
    main()