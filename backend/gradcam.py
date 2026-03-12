import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils import numpy_to_base64, pil_to_base64


class GradCAM:
    """
    Grad-CAM for the UltrasoundModel.
    Hooks into the last conv block to generate heatmaps.
    """
    def __init__(self, model):
        self.model  = model
        self.grads  = None
        self.acts   = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.features[-1]   # last conv block

        def save_acts(module, input, output):
            self.acts = output

        def save_grads(module, grad_in, grad_out):
            self.grads = grad_out[0]

        self._hooks.append(target.register_forward_hook(save_acts))
        self._hooks.append(target.register_full_backward_hook(save_grads))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, class_idx):
        """
        Args:
            input_tensor : (1, 3, 224, 224)
            class_idx    : index of predicted class
        Returns:
            heatmap_rgb  : numpy (224, 224, 3)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        score  = logits[0, class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Pool gradients → weights
        weights = self.grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().detach().numpy()

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = (cam * 255).astype(np.uint8)

        # Resize + colormap
        cam_resized  = cv2.resize(cam, (224, 224))
        heatmap      = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap_rgb  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap_rgb


def overlay_heatmap(pil_image, heatmap_rgb, alpha=0.5):
    """Blend original image with Grad-CAM heatmap."""
    orig    = np.array(pil_image.resize((224, 224))).astype(np.float32)
    heat    = heatmap_rgb.astype(np.float32)
    blended = (alpha * orig + (1 - alpha) * heat).clip(0, 255).astype(np.uint8)
    return numpy_to_base64(blended)


def generate_heatmap(model, image_tensor, pil_image, class_idx):
    """
    Full pipeline: generate Grad-CAM and overlay on original image.
    Returns base64 string.
    """
    try:
        cam_gen  = GradCAM(model)
        heatmap  = cam_gen.generate(image_tensor, class_idx=class_idx)
        overlay  = overlay_heatmap(pil_image, heatmap)
        cam_gen.remove_hooks()
        return overlay
    except Exception as e:
        print(f"⚠️  Grad-CAM failed: {e}")
        return pil_to_base64(pil_image.resize((224, 224)))
