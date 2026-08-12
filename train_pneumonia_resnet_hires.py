"""
Pneumonia classifier v4 -- same ResNet18 transfer-learning architecture as
train_pneumonia_resnet.py, but trained on the ORIGINAL Kermany et al. chest
X-ray JPEGs (roughly 1000px) instead of the 28x28 PneumoniaMNIST downsample.

Why: the booth game shows the visitor exactly the image the model saw and the
image Grad-CAM was computed on, so tap-hitbox accuracy depends on that image
being real detail, not a blown-up 28px thumbnail. Training directly on
full-resolution images means the demo can show genuinely sharp X-rays with a
heatmap that's still pixel-aligned to what's on screen.

The official Kaggle split's "val" folder is only 16 images (8/8) -- too small
and noisy to trust for model selection -- so train+val are pooled here and a
fresh stratified 90/10 split is carved out. The official "test" folder (624
images) is left untouched so results stay comparable to published benchmarks
on this dataset.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from train_pneumonia_resnet import PneumoniaResNet, to_3ch_normalized, IMAGENET_MEAN, IMAGENET_STD

DATA_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\data\chest_xray"
OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
IMG_SIZE = 224
VAL_FRACTION = 0.1
# per-layer learning rates (FC_LR / BACKBONE_LR) live on PneumoniaResNet.param_groups()
# in train_pneumonia_resnet.py -- reused as-is here, not redefined


class PathListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def stratified_split_samples(samples, val_fraction, seed):
    by_class = {}
    for idx, (_, label) in enumerate(samples):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = to_3ch_normalized(IMG_SIZE)
    return train_transform, eval_transform


def get_loaders():
    train_transform, eval_transform = build_transforms()

    train_folder = ImageFolder(os.path.join(DATA_DIR, "train"))
    val_folder = ImageFolder(os.path.join(DATA_DIR, "val"))
    assert train_folder.classes == ["NORMAL", "PNEUMONIA"], train_folder.classes
    assert val_folder.classes == train_folder.classes

    pooled_samples = train_folder.samples + val_folder.samples
    train_idx, val_idx = stratified_split_samples(pooled_samples, VAL_FRACTION, SEED)

    train_ds = PathListDataset([pooled_samples[i] for i in train_idx], train_transform)
    val_ds = PathListDataset([pooled_samples[i] for i in val_idx], eval_transform)
    test_ds = ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)
    assert test_ds.classes == train_folder.classes

    train_labels = [pooled_samples[i][1] for i in train_idx]

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        train_labels,
    )


def compute_pos_weight(train_labels):
    labels = np.array(train_labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, test_loader, train_labels = get_loaders()
    print(f"train={len(train_labels)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    pos_weight = compute_pos_weight(train_labels).to(device)
    print(f"pos_weight={pos_weight.item():.3f}")

    model = PneumoniaResNet().to(device)
    optimizer = torch.optim.Adam(model.param_groups(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()

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
            torch.save(best_state, os.path.join(OUT_DIR, "pneumonia_resnet_hires.pt"))
            print(f"  -> new best, saved (val_acc={val_acc:.4f})")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"\ntest_acc={test_acc:.4f} (best val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "pneumonia_resnet_hires.pt"))
    print(f"saved -> {os.path.join(OUT_DIR, 'pneumonia_resnet_hires.pt')}")


if __name__ == "__main__":
    main()
