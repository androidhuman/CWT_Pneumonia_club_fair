# 명의 vs AI — Chest X-ray Pneumonia Detection Booth Demo

Club-fair booth demo reproducing the pipeline structure of Sharma et al.,
*"Deep learning models for tuberculosis detection and infected region
visualization in chest X-ray images"* (Intelligent Medicine, 2024), adapted
from tuberculosis to **pneumonia** detection: segmentation → classification →
Grad-CAM visualization.

Visitors look at a chest X-ray, guess normal vs. pneumonia, tap where they
think the abnormality is, then see the AI's Grad-CAM heatmap and get scored
on both diagnostic accuracy and how closely their tap matches where the
model actually focused.

## Pipeline

| Stage | Method | Result |
|---|---|---|
| 1. Lung segmentation | U-Net, trained on Montgomery + Shenzhen CXR/mask pairs (704 images) | test Dice 0.954, Jaccard 0.913, pixel accuracy 97.7% |
| 2. Pneumonia classification | ResNet18 (ImageNet-pretrained, `layer4`+`fc` fine-tuned) on PneumoniaMNIST | test accuracy 90.9% |
| 3. Explainability | Grad-CAM on the classifier's last conv block | — |

Stage 1 uses a different image domain (Montgomery/Shenzhen, high-res adult
CXRs with lung masks) than stages 2–3 (PneumoniaMNIST, 28×28, no masks), so
it isn't wired into the live classifier — it's shown as a static pipeline
walkthrough in the app's intro screen, while the interactive game exercises
stages 2–3.

The classifier's val/test accuracy gap (~97% val vs. ~91% test) reflects a
real property of the underlying Kermany pneumonia dataset: train/val and
test images come from different patient batches, i.e. genuine distribution
shift rather than pure overfitting.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

All inference is precomputed offline (`prepare_demo_samples.py`,
`prepare_unet_examples.py`) so the deployed app only reads static
images/JSON — no model inference at request time.

## Retraining

- `train_unet.py` — lung segmentation U-Net (needs the Kaggle ["Chest X-ray
  masks and labels"](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
  dataset locally under `data/Lung Segmentation/`, not included in this repo)
- `train_pneumonia_resnet.py` — pneumonia classifier (ResNet18 transfer
  learning; downloads PneumoniaMNIST automatically)
- `train_pneumonia_cnn.py` — earlier from-scratch CNN baseline, kept as a
  fallback (test accuracy 88.0%)
- `gradcam.py` — Grad-CAM implementation
- `prepare_demo_samples.py` / `prepare_unet_examples.py` — regenerate the
  static assets the app serves, after retraining
