"""
Pneumonia classifier (CNN) on PneumoniaMNIST, with a Grad-CAM-ready architecture
(named last conv layer so hooks can be attached cleanly in gradcam.py).

v2 (2026-08-11): first pass (no BatchNorm, no augmentation, no class weighting) got
val_acc=92.9% / test_acc=88.0% -- the val/test gap partly reflects a known property
of the underlying Kermany pneumonia dataset (train/val and test come from different
patient batches, i.e. real distribution shift, not pure overfitting), but the model
itself was under-regularized. Added: BatchNorm, dropout, light augmentation, pos_weight
for the 74/26 class imbalance (train: 3494 pneumonia vs 1214 normal), LR scheduling.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST

OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 14->7
        )
        # named separately (not folded into `features`) so Grad-CAM can hook this
        # exact layer's output/gradient by name -- "the last conv layer before pooling"
        self.last_conv = nn.Conv2d(32, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        # inplace=False here (unlike the ReLUs in self.features): Grad-CAM's backward
        # hook on last_conv wraps its output, and an in-place op right after would
        # corrupt that wrapped view -- PyTorch refuses this combination at runtime.
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(self.bn(self.last_conv(x)))
        pooled = self.pool(x).flatten(1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(-1)


def get_loaders():
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor()])

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

    model = PneumoniaCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
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
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}  train_loss={epoch_loss / n_batches:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"\ntest_acc={test_acc:.4f} (best val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "pneumonia_cnn.pt"))
    print(f"saved -> {os.path.join(OUT_DIR, 'pneumonia_cnn.pt')}")


if __name__ == "__main__":
    main()
