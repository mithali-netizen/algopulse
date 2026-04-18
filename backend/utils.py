import io
import base64
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

TARGET_SIZE = 224


def resize_and_pad(pil_img, size=TARGET_SIZE, fill=(0, 0, 0)):
    """
    Preserve the whole image by resizing to fit within a square canvas and
    padding the remaining area instead of cropping.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    width, height = pil_img.size
    if width == 0 or height == 0:
        raise ValueError("Invalid image size")

    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = pil_img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    pad_right = size - new_width - pad_left
    pad_bottom = size - new_height - pad_top

    return ImageOps.expand(
        resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=fill,
    )


def get_transform(augment=False):
    """
    Shared preprocessing for training and inference:
    - preserve aspect ratio
    - avoid cropping by padding to a fixed square
    - convert to tensor and normalize
    """
    ops = [
        transforms.Lambda(lambda img: resize_and_pad(img, TARGET_SIZE)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(contrast=0.3, brightness=0.2),
        ]
    ops += [
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
      - tensor  : shape (1, 3, 224, 224) without cropping source content
      - pil_img : original PIL image for Grad-CAM overlay
      - size    : (W, H) tuple of ORIGINAL image (for UI/debug)
    """
    pil_img  = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = get_transform(augment=False)
    tensor = transform(pil_img).unsqueeze(0)          # → (1, 3, 224, 224)
    w, h = pil_img.size
    return tensor, pil_img, (w, h)


def pil_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def numpy_to_base64(np_array):
    pil_img = Image.fromarray(np_array.astype(np.uint8))
    return pil_to_base64(pil_img)
