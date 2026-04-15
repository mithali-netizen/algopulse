import os
import torch
import torch.nn as nn
from torchvision import models

# ── Labels ────────────────────────────────────────────────────────────────────
CLASS_NAMES          = ["Benign", "Malignant", "Normal"]
CONFIDENCE_THRESHOLD = 0.75


# ── Adaptive EfficientNet branch ──────────────────────────────────────────────
class UltrasoundModel(nn.Module):
    """
    EfficientNet-B0 with:
    - AdaptiveAvgPool2d → works on ANY input size, no cropping needed
    - Custom classifier head
    """
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze early layers
        for param in list(base.parameters())[:-20]:
            param.requires_grad = False

        self.features = base.features

        # AdaptiveAvgPool2d(1) collapses any spatial size → (batch, C, 1, 1)
        # This is what makes the model size-agnostic
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x can be any spatial size — no fixed 224x224 required
        x = self.features(x)      # → (batch, 1280, H', W')  H'/W' varies
        x = self.pool(x)          # → (batch, 1280, 1, 1)
        x = x.flatten(1)          # → (batch, 1280)
        x = self.classifier(x)    # → (batch, num_classes)
        return x


# ── Load model ────────────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        _model = UltrasoundModel(num_classes=3)
        if os.path.exists("best_model.pth"):
            _model.load_state_dict(
                torch.load("best_model.pth", map_location="cpu")
            )
            print("✅ Loaded trained weights from best_model.pth")
        else:
            print("⚠️  No trained weights found — using ImageNet pretrained weights")
        _model.eval()
    return _model


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_tensor):
    """
    Args:
        image_tensor : (1, 3, H, W) — any size
    Returns:
        label, confidence, probs dict, flagged bool
    """
    model = get_model()
    with torch.no_grad():
        logits       = model(image_tensor)
        probs_tensor = torch.softmax(logits, dim=1)[0]

    probs      = {CLASS_NAMES[i]: float(probs_tensor[i]) for i in range(3)}
    best_idx   = int(torch.argmax(probs_tensor))
    confidence = float(probs_tensor[best_idx])
    label      = CLASS_NAMES[best_idx]
    flagged    = confidence < CONFIDENCE_THRESHOLD

    return label, confidence, probs, flagged


# ── Get embedding for similarity search ────────────────────────────────────────
def get_embedding(image_tensor):
    """
    Extract 1280-dimensional embedding from the model.
    Args:
        image_tensor: (1, 3, H, W) preprocessed tensor
    Returns:
        numpy array of shape (1280,)
    """
    model = get_model()
    with torch.no_grad():
        x = model.features(image_tensor)
        x = model.pool(x)
        embedding = x.flatten(1).squeeze(0).numpy()
    return embedding