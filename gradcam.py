"""
Grad-CAM for PneumoniaResNet -- works with either checkpoint (train_pneumonia_resnet.py's
28px-trained weights or train_pneumonia_resnet_hires.py's full-resolution weights),
since the architecture and last_conv hook point are identical either way.

Hooks model.last_conv to capture (a) its output activations on the forward pass and
(b) the gradient flowing into that output on the backward pass. Grad-CAM's core idea:
weight each activation channel by how much increasing it would increase the predicted
class score (that's exactly what the globally-averaged gradient measures), then take
a weighted sum of channels -- channels the model relied on heavily for THIS prediction
light up, channels it ignored don't.
"""
import numpy as np
import torch
import torch.nn.functional as F

from train_pneumonia_resnet import PneumoniaResNet


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        model.last_conv.register_forward_hook(self._save_activations)
        model.last_conv.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor):
        """image_tensor: (1, 1, H, W). Returns (heatmap as (H,W) numpy in [0,1], prob float)."""
        self.model.eval()
        image_tensor = image_tensor.clone().requires_grad_(False)

        logit = self.model(image_tensor)  # (1,)
        prob = torch.sigmoid(logit).item()

        self.model.zero_grad()
        logit.backward()  # backprop from the single pneumonia-confidence logit

        # global-average-pool gradients over spatial dims -> one importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only care about features that push TOWARD this class, not away

        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize to [0,1] for display

        return cam, prob


def load_model(checkpoint_path, device="cpu"):
    model = PneumoniaResNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    from PIL import Image
    from train_pneumonia_resnet import to_3ch_normalized
    from train_pneumonia_resnet_hires import DATA_DIR, IMG_SIZE
    import glob
    import os

    ckpt = r"C:\Users\jihunkwon\PycharmProjects\project_project\CWT_pneumonia\pneumonia_resnet_hires.pt"
    model = load_model(ckpt)
    cam_tool = GradCAM(model)

    sample_path = glob.glob(os.path.join(DATA_DIR, "test", "PNEUMONIA", "*.jpeg"))[0]
    img = to_3ch_normalized(IMG_SIZE)(Image.open(sample_path).convert("RGB"))
    heatmap, prob = cam_tool(img.unsqueeze(0))
    print("file:", sample_path, "predicted prob:", prob, "heatmap shape:", heatmap.shape)
