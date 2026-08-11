"""
Lung segmentation U-Net training.

Pipeline: list_all_stems -> split -> LungSegmentationDataset (preprocessor.py) ->
DataLoader -> UNet -> train with BCE+Dice loss -> evaluate with Dice/Jaccard/accuracy,
matching the metrics the reference paper (Sharma et al. 2024) reported.
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessor import list_all_stems, LungSegmentationDataset

BASE = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\data\Lung Segmentation"
CXR_DIR = os.path.join(BASE, "CXR_png")
MASK_DIR = os.path.join(BASE, "masks")
OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia"

SEED = 42
IMG_SIZE = 128  # 224 like the paper is heavier on CPU; 128 trains much faster and is
                # plenty for a first working pass -- bump to 224 later if accuracy matters more than speed
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-3
BASE_CHANNELS = 16  # paper's UNet starts at 64; 16 is a much lighter/faster CPU-friendly version


def split_stems(stems, ratios=(0.8, 0.1, 0.1), seed=SEED):
    stems = stems[:]
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return stems[:n_train], stems[n_train:n_train + n_val], stems[n_train + n_val:]


class DoubleConv(nn.Module):
    """conv -> ReLU -> conv -> ReLU, the basic repeated block in every U-Net stage."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=BASE_CHANNELS):
        super().__init__()
        # encoder: each stage halves spatial size, doubles channels
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        # decoder: each stage upsamples, then concatenates the matching encoder
        # feature map (skip connection) before another DoubleConv
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)  # *16 because concat doubles channels
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)  # 1x1 conv -> per-pixel logit

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # skip connection
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # logits, shape (batch, 1, H, W)


def dice_coeff(pred, target, eps=1e-6):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    return ((2 * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)).mean()


def jaccard_index(pred, target, eps=1e-6):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    return 1 - dice_coeff(probs, target)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, jaccards, accs = [], [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        dices.append(dice_coeff(preds, masks).item())
        jaccards.append(jaccard_index(preds, masks).item())
        accs.append((preds == masks).float().mean().item())
    return {
        "dice": sum(dices) / len(dices),
        "jaccard": sum(jaccards) / len(jaccards),
        "accuracy": sum(accs) / len(accs),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    stems = list_all_stems(MASK_DIR)
    train_stems, val_stems, test_stems = split_stems(stems)
    print(f"train={len(train_stems)} val={len(val_stems)} test={len(test_stems)}")

    train_ds = LungSegmentationDataset(CXR_DIR, MASK_DIR, train_stems, img_size=IMG_SIZE)
    val_ds = LungSegmentationDataset(CXR_DIR, MASK_DIR, val_stems, img_size=IMG_SIZE)
    test_ds = LungSegmentationDataset(CXR_DIR, MASK_DIR, test_stems, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            # combined BCE + Dice loss: BCE is stable per-pixel classification,
            # Dice directly optimizes the overlap metric we actually evaluate on
            loss = bce(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d}  train_loss={epoch_loss / n_batches:.4f}  "
              f"val_dice={val_metrics['dice']:.4f}  val_jaccard={val_metrics['jaccard']:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print("\n=== test set (best val checkpoint) ===")
    print(f"dice={test_metrics['dice']:.4f}  jaccard={test_metrics['jaccard']:.4f}  "
          f"accuracy={test_metrics['accuracy']:.4f}")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "unet_lung_seg.pt"))
    print(f"\nsaved model -> {os.path.join(OUT_DIR, 'unet_lung_seg.pt')}")


if __name__ == "__main__":
    main()
