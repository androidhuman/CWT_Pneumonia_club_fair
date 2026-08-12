"""
Same job as prepare_demo_samples.py, but sourced from the full-resolution Kermany
chest X-ray test set (via train_pneumonia_resnet_hires.py's model/pipeline)
instead of 28x28 PneumoniaMNIST -- so the displayed image, the model's input,
and the Grad-CAM heatmap are all pixel-aligned at real resolution.
"""
import glob
import json
import os

import numpy as np
from PIL import Image

from gradcam import GradCAM, load_model
from train_pneumonia_resnet_hires import DATA_DIR, IMG_SIZE
from train_pneumonia_resnet import to_3ch_normalized

OUT_DIR = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\demo_samples_hires"
CKPT = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\pneumonia_resnet_hires.pt"
N_PER_CLASS = 15
DISPLAY_SIZE = 320
HOT_THRESHOLD = 0.6


def colorize_heatmap(heatmap):
    r = np.clip(heatmap * 2, 0, 1)
    g = np.clip(heatmap * 2 - 0.5, 0, 1)
    b = np.clip(heatmap * 2 - 1.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(gray_img_arr, heatmap, alpha=0.45):
    base_rgb = np.stack([gray_img_arr] * 3, axis=-1).astype(np.float32)
    hm_rgb = colorize_heatmap(heatmap).astype(np.float32)
    blended = (1 - alpha) * base_rgb + alpha * hm_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def list_test_files():
    files = []
    for label_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for path in glob.glob(os.path.join(DATA_DIR, "test", label_name, ext)):
                files.append((path, label))
    return files


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(CKPT)
    cam_tool = GradCAM(model)
    model_transform = to_3ch_normalized(IMG_SIZE)

    files = list_test_files()
    normal_files = [f for f in files if f[1] == 0]
    pneumonia_files = [f for f in files if f[1] == 1]

    rng = np.random.default_rng(42)
    rng.shuffle(normal_files)
    rng.shuffle(pneumonia_files)

    # Only AI-correct predictions make the demo: a booth visitor comparing their tap
    # against Grad-CAM shouldn't also have to untangle "the AI itself was wrong here" in
    # a few seconds. The model's real ~90% test accuracy is still stated on the intro
    # screen -- this filter only curates which 30 examples get shown, it doesn't hide
    # the overall number.
    manifest = []
    n_seen = {"normal": 0, "pneumonia": 0}
    n_found = {"normal": 0, "pneumonia": 0}
    for label_name, candidates in [("normal", normal_files), ("pneumonia", pneumonia_files)]:
        for path, label in candidates:
            if n_found[label_name] >= N_PER_CLASS:
                break
            n_seen[label_name] += 1
            raw = Image.open(path).convert("RGB")

            model_input = model_transform(raw).unsqueeze(0)
            heatmap, prob = cam_tool(model_input)  # (IMG_SIZE, IMG_SIZE)
            predicted_label = "pneumonia" if prob > 0.5 else "normal"
            if predicted_label != label_name:
                continue  # AI got this one wrong -- skip, don't use it in the demo

            i = len(manifest)
            # same resize the model saw, kept as grayscale for display so heatmap stays aligned
            gray_arr = np.array(raw.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.uint8)
            overlay_arr = overlay(gray_arr, heatmap)

            plain_img = Image.fromarray(gray_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            overlay_img = Image.fromarray(overlay_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)

            plain_path = os.path.join(OUT_DIR, f"sample_{i:02d}_plain.png")
            overlay_path = os.path.join(OUT_DIR, f"sample_{i:02d}_overlay.png")
            plain_img.save(plain_path)
            overlay_img.save(overlay_path)

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
                "true_label": label_name,
                "predicted_prob": round(float(prob), 4),
                "predicted_label": predicted_label,
                "correct": True,
            })
            n_found[label_name] += 1

    rng.shuffle(manifest)
    for i, m in enumerate(manifest):
        m["id"] = i

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"{len(manifest)}개 샘플 생성 완료 (전부 AI 정답), "
          f"정상 {n_seen['normal']}장/폐렴 {n_seen['pneumonia']}장 훑어봄")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
