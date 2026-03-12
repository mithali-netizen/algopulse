import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils import numpy_to_base64, pil_to_base64


class GradCAM:
    """
    Grad-CAM that works on any input size.
    Heatmap is resized to match the ORIGINAL image dimensions.
    """
    def __init__(self, model):
        self.model  = model
        self.grads  = None
        self.acts   = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.features[-1]

        def save_acts(module, input, output):
            self.acts = output

        def save_grads(module, grad_in, grad_out):
            self.grads = grad_out[0]

        self._hooks.append(target.register_forward_hook(save_acts))
        self._hooks.append(target.register_full_backward_hook(save_grads))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, class_idx, out_size):
        """
        Args:
            input_tensor : (1, 3, H, W) — any size
            class_idx    : predicted class index
            out_size     : (W, H) of the original image to resize heatmap to
        Returns:
            heatmap_rgb  : numpy (out_H, out_W, 3)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        score  = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Weights = global average of gradients
        weights = self.grads.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().detach().numpy()

        # Handle edge case where cam is scalar
        if cam.ndim == 0:
            cam = np.zeros((7, 7))

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = (cam * 255).astype(np.uint8)

        # ── Resize to ORIGINAL image size (no fixed 224!) ──────────────────
        out_w, out_h = out_size
        cam_resized  = cv2.resize(cam, (out_w, out_h),
                                  interpolation=cv2.INTER_LINEAR)
        heatmap      = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap_rgb  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap_rgb


def overlay_heatmap(pil_image, heatmap_rgb, alpha=0.5):
    """
    Blend original PIL image with heatmap.
    Both are already the same size — no additional resizing.
    """
    orig    = np.array(pil_image).astype(np.float32)

    # Ensure heatmap matches original exactly
    if orig.shape[:2] != heatmap_rgb.shape[:2]:
        h, w = orig.shape[:2]
        heatmap_rgb = cv2.resize(heatmap_rgb, (w, h))

    heat    = heatmap_rgb.astype(np.float32)
    blended = (alpha * orig + (1 - alpha) * heat).clip(0, 255).astype(np.uint8)
    return numpy_to_base64(blended)


def generate_heatmap(model, image_tensor, pil_image, class_idx):
    """
    Full pipeline: Grad-CAM → overlay on original image at original size.
    Returns base64 string.
    """
    try:
        orig_w, orig_h = pil_image.size   # original dimensions
        cam_gen  = GradCAM(model)
        heatmap  = cam_gen.generate(image_tensor, class_idx,
                                    out_size=(orig_w, orig_h))
        overlay  = overlay_heatmap(pil_image, heatmap)
        cam_gen.remove_hooks()
        return overlay
    except Exception as e:
        print(f"⚠️  Grad-CAM failed: {e}")
        return pil_to_base64(pil_image)