import io
import base64
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

IMAGE_SIZE = 224

def get_transform():
    """
    Preprocessing pipeline for ultrasound images.
    Includes noise reduction via normalization.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # ultrasound is grayscale → convert to 3ch for model
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(file_bytes):
    """
    Takes raw uploaded file bytes and returns:
      - tensor  : shape (1, 3, 224, 224)  ready for the model
      - pil_img : original PIL image      used for Grad-CAM overlay
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = get_transform()
    tensor = transform(pil_img).unsqueeze(0)   # add batch dimension → (1,3,224,224)
    return tensor, pil_img


def pil_to_base64(pil_img):
    """Convert a PIL image to a base64 string so React can display it."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def numpy_to_base64(np_array):
    """Convert a numpy uint8 array (H,W,3) to a base64 string."""
    pil_img = Image.fromarray(np_array.astype(np.uint8))
    return pil_to_base64(pil_img)
