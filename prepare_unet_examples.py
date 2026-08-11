"""
Precompute a few U-Net lung-segmentation examples (CXR + predicted mask overlay) for
the Streamlit app's pipeline-intro screen. This stage isn't part of the interactive
tap game (different image domain than the PneumoniaMNIST classifier -- see app design
notes), it's shown as a static "here's stage 1 of the pipeline" demonstration.
"""
import os

import numpy as np
from PIL import Image
import torch

from preprocessor import list_all_stems, LungSegmentationDataset
from train_unet import UNet, split_stems, dice_coeff, IMG_SIZE

BASE = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\data\Lung Segmentation"
CXR_DIR = os.path.join(BASE, "CXR_png")
MASK_DIR = os.path.join(BASE, "masks")
CKPT = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\unet_lung_seg.pt"
OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\unet_examples"
N_EXAMPLES = 4
DISPLAY_SIZE = 320


def overlay_mask(gray_arr, mask_arr, color=(66, 220, 90), alpha=0.35):
    """gray_arr, mask_arr: (H,W) in [0,255]/[0,1]. Green-tint the predicted lung region."""
    base_rgb = np.stack([gray_arr] * 3, axis=-1).astype(np.float32)
    tint = np.zeros_like(base_rgb)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    m = mask_arr[..., None]
    blended = base_rgb * (1 - alpha * m) + tint * (alpha * m)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    stems = list_all_stems(MASK_DIR)
    _, _, test_stems = split_stems(stems)  # same seed as training -> untouched-by-training images
    chosen = test_stems[:N_EXAMPLES]

    ds = LungSegmentationDataset(CXR_DIR, MASK_DIR, chosen, img_size=IMG_SIZE)

    examples = []
    with torch.no_grad():
        for i, (image_tensor, mask_tensor) in enumerate(ds):
            logit = model(image_tensor.unsqueeze(0))
            pred_mask = (torch.sigmoid(logit) > 0.5).float()
            dice = dice_coeff(pred_mask, mask_tensor.unsqueeze(0)).item()

            gray_arr = (image_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
            pred_arr = pred_mask.squeeze().numpy()

            overlay_arr = overlay_mask(gray_arr, pred_arr)
            overlay_img = Image.fromarray(overlay_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            plain_img = Image.fromarray(gray_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)

            overlay_path = os.path.join(OUT_DIR, f"unet_{i:02d}_overlay.png")
            plain_path = os.path.join(OUT_DIR, f"unet_{i:02d}_plain.png")
            overlay_img.save(overlay_path)
            plain_img.save(plain_path)

            examples.append({
                "plain": os.path.basename(plain_path),
                "overlay": os.path.basename(overlay_path),
                "dice": round(dice, 4),
            })
            print(f"{chosen[i]}: dice={dice:.4f}")

    import json
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"saved {len(examples)} examples -> {OUT_DIR}")


if __name__ == "__main__":
    main()
