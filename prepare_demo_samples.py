"""
Precompute everything the booth game needs for a fixed set of demo images: prediction,
Grad-CAM heatmap (as a displayable overlay PNG), and the "AI attention region" used to
score whether the visitor tapped where the model actually looked. All done offline so
the deployed app never runs live inference -- just looks up precomputed results.
"""
import json
import os

import numpy as np
from PIL import Image
import torch
from medmnist import PneumoniaMNIST

from gradcam import GradCAM, load_model
from train_pneumonia_resnet import to_3ch_normalized, IMG_SIZE

OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\demo_samples"
CKPT = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\pneumonia_resnet.pt"
N_PER_CLASS = 15  # 15 normal + 15 pneumonia = 30 total, randomly served during the game
DISPLAY_SIZE = 280  # upscaled from IMG_SIZE for visibility on a phone screen
HOT_THRESHOLD = 0.6  # heatmap pixels above this (post-normalization) count as "AI focus area"


def colorize_heatmap(heatmap):
    """heatmap: (28,28) in [0,1] -> (28,28,3) uint8 using a simple red-hot colormap
    (avoids a matplotlib dependency for just this)."""
    r = np.clip(heatmap * 2, 0, 1)
    g = np.clip(heatmap * 2 - 0.5, 0, 1)
    b = np.clip(heatmap * 2 - 1.0, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay(gray_img_arr, heatmap, alpha=0.45):
    """gray_img_arr: (28,28) uint8, heatmap: (28,28) in [0,1]"""
    base_rgb = np.stack([gray_img_arr] * 3, axis=-1).astype(np.float32)
    hm_rgb = colorize_heatmap(heatmap).astype(np.float32)
    blended = (1 - alpha) * base_rgb + alpha * hm_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(CKPT)
    cam_tool = GradCAM(model)

    ds = PneumoniaMNIST(split="test", download=True, size=28)  # no transform -> raw PIL images
    labels = np.array([ds[i][1][0] for i in range(len(ds))])
    model_transform = to_3ch_normalized(IMG_SIZE)

    normal_idx = np.where(labels == 0)[0]
    pneumonia_idx = np.where(labels == 1)[0]
    rng = np.random.default_rng(42)
    chosen = np.concatenate([
        rng.choice(normal_idx, size=min(N_PER_CLASS, len(normal_idx)), replace=False),
        rng.choice(pneumonia_idx, size=min(N_PER_CLASS, len(pneumonia_idx)), replace=False),
    ])
    rng.shuffle(chosen)

    manifest = []
    for i, idx in enumerate(chosen):
        pil_img, label = ds[int(idx)]  # PIL grayscale image, native 28x28

        model_input = model_transform(pil_img).unsqueeze(0)  # (1,3,IMG_SIZE,IMG_SIZE), normalized
        heatmap, prob = cam_tool(model_input)  # (IMG_SIZE, IMG_SIZE)

        # display base at the same resolution as the heatmap so overlay pixels line up
        gray_arr = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
        overlay_arr = overlay(gray_arr, heatmap)

        plain_img = Image.fromarray(gray_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
        overlay_img = Image.fromarray(overlay_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)

        plain_path = os.path.join(OUT_DIR, f"sample_{i:02d}_plain.png")
        overlay_path = os.path.join(OUT_DIR, f"sample_{i:02d}_overlay.png")
        plain_img.save(plain_path)
        overlay_img.save(overlay_path)

        # "hot region" mask at display resolution, used later to score visitor taps
        hot_mask = (heatmap >= HOT_THRESHOLD)
        hot_mask_img = Image.fromarray((hot_mask * 255).astype(np.uint8)).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST
        )
        hot_mask_path = os.path.join(OUT_DIR, f"sample_{i:02d}_hotmask.png")
        hot_mask_img.save(hot_mask_path)

        manifest.append({
            "id": i,
            "plain_image": os.path.basename(plain_path),
            "overlay_image": os.path.basename(overlay_path),
            "hot_mask": os.path.basename(hot_mask_path),
            "true_label": "pneumonia" if int(label[0]) == 1 else "normal",
            "predicted_prob": round(float(prob), 4),
            "predicted_label": "pneumonia" if prob > 0.5 else "normal",
            "correct": (prob > 0.5) == (int(label[0]) == 1),
        })

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_correct = sum(m["correct"] for m in manifest)
    print(f"{len(manifest)}개 샘플 생성 완료, 모델이 맞춘 개수: {n_correct}/{len(manifest)}")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
