import io
import base64
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ── Dynamic size — no cropping, just resize to nearest multiple of 32 ─────────
def get_dynamic_size(pil_img, max_size=512, min_size=224):
    """
    Returns (width, height) that:
    - Keeps original aspect ratio
    - Fits within max_size x max_size
    - Is at least min_size on the shorter side
    - Each dimension is a multiple of 32 (required by EfficientNet)
    """
    w, h = pil_img.size
    scale = min(max_size / w, max_size / h)
    if min(w, h) * scale < min_size:
        scale = min_size / min(w, h)
    new_w = max(32, int(w * scale) // 32 * 32)
    new_h = max(32, int(h * scale) // 32 * 32)
    return new_w, new_h


def get_transform(size=None):
    """
    If size is given → resize to that exact (w,h)
    If size is None  → just normalize, no resize (use after manual resize)
    """
    ops = []
    if size is not None:
        ops.append(transforms.Resize(size, antialias=True))   # no crop!
    ops += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return transforms.Compose(ops)


def preprocess_image(file_bytes):
    """
    Takes raw uploaded file bytes.
    Returns:
      - tensor  : shape (1, 3, H, W) — dynamic size, no cropping
      - pil_img : original PIL image for Grad-CAM overlay
      - size    : (W, H) tuple used for this image
    """
    pil_img  = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    new_w, new_h = get_dynamic_size(pil_img)
    transform = get_transform(size=(new_h, new_w))   # transforms.Resize takes (H, W)
    tensor = transform(pil_img).unsqueeze(0)          # → (1, 3, H, W)
    return tensor, pil_img, (new_w, new_h)


def pil_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def numpy_to_base64(np_array):
    pil_img = Image.fromarray(np_array.astype(np.uint8))
    return pil_to_base64(pil_img)