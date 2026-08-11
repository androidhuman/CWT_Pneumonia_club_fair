"""
Pneumonia classifier v3 -- transfer learning with an ImageNet-pretrained ResNet18.

Why: v1/v2 trained a small CNN from scratch and topped out at test_acc ~88-89%
(best val_acc 92.9% / 96.8% respectively). That val/test gap is a real property of
the underlying Kermany dataset (train+val and test come from different patient
batches -- genuine distribution shift, not just overfitting), and a from-scratch
CNN has no prior to fall back on when the input distribution shifts slightly.

This is also more faithful to the reference paper (Sharma et al. 2024), which
classifies with a pretrained Xception, not a from-scratch CNN. Here: ResNet18
pretrained on ImageNet, grayscale images replicated to 3 channels (keeps the
pretrained conv1 filters meaningful, no reinit needed), early layers frozen,
layer4 + fc fine-tuned at a small LR.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import PneumoniaMNIST

OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
IMG_SIZE = 96
FC_LR = 1e-3
BACKBONE_LR = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PneumoniaResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # Grad-CAM hooks this -- last conv feature maps before pooling
        self.avgpool = backbone.avgpool
        self.classifier = nn.Linear(backbone.fc.in_features, 1)

        # named the same way as the v2 model so gradcam.py's hook-by-attribute
        # keeps working unchanged
        self.last_conv = self.layer4

        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False
        for p in self.layer3.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x).squeeze(-1)

    def param_groups(self):
        return [
            {"params": self.layer4.parameters(), "lr": BACKBONE_LR},
            {"params": self.classifier.parameters(), "lr": FC_LR},
        ]


def to_3ch_normalized(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_loaders():
    # PneumoniaMNIST yields PIL images at size=28 by default when transform is applied after;
    # medmnist's `size=28` param controls the raw image resolution served, transform handles resize/aug.
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = to_3ch_normalized(IMG_SIZE)

    train_ds = PneumoniaMNIST(split="train", download=True, size=28, transform=train_transform)
    val_ds = PneumoniaMNIST(split="val", download=True, size=28, transform=eval_transform)
    test_ds = PneumoniaMNIST(split="test", download=True, size=28, transform=eval_transform)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        train_ds,
    )


def compute_pos_weight(train_ds):
    labels = np.array([train_ds[i][1][0] for i in range(len(train_ds))])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().squeeze(-1)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, test_loader, train_ds_for_weight = get_loaders()
    pos_weight = compute_pos_weight(train_ds_for_weight).to(device)
    print(f"pos_weight={pos_weight.item():.3f}")

    model = PneumoniaResNet().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.Adam(model.param_groups(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().squeeze(-1)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[1]["lr"]
        print(f"epoch {epoch:02d}  train_loss={epoch_loss / n_batches:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"\ntest_acc={test_acc:.4f} (best val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "pneumonia_resnet.pt"))
    print(f"saved -> {os.path.join(OUT_DIR, 'pneumonia_resnet.pt')}")


if __name__ == "__main__":
    main()
