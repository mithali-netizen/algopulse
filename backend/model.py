import torch
import torch.nn as nn
from torchvision import models

# ── Labels ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
CONFIDENCE_THRESHOLD = 0.75     # below this → flag for clinical review

# ── Single branch EfficientNet-B0 model ─────────────────────────────────────
class UltrasoundModel(nn.Module):
    """
    Single EfficientNet-B0 backbone for ultrasound classification.
    Classifies into: Benign / Malignant / Normal
    Uses transfer learning from ImageNet pretrained weights.
    """
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        # Load pretrained EfficientNet-B0
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze early layers (keep ImageNet knowledge)
        for param in list(base.parameters())[:-20]:
            param.requires_grad = False

        # Keep feature extractor
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)   # → (batch, 1280, 1, 1)

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Attention map storage (for Grad-CAM)
        self.last_conv_output = None

    def forward(self, x):
        x = self.features(x)
        self.last_conv_output = x          # save for Grad-CAM
        x = self.pool(x)
        x = x.flatten(1)                   # → (batch, 1280)
        x = self.classifier(x)
        return x


# ── Helper: load model once ──────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        _model = UltrasoundModel(num_classes=3)
        _model.eval()
        print("✅ UltrasoundModel loaded with EfficientNet-B0 pretrained weights")
    return _model


# ── Helper: run inference ────────────────────────────────────────────────────
def predict(image_tensor):
    """
    Args:
        image_tensor : shape (1, 3, 224, 224)
    Returns:
        label        : str   e.g. "Malignant"
        confidence   : float (0-1)
        probs        : dict  e.g. {"Benign": 0.1, "Malignant": 0.85, "Normal": 0.05}
        flagged      : bool  True if confidence < threshold
    """
    model = get_model()

    with torch.no_grad():
        logits = model(image_tensor)                      # (1, 3)
        probs_tensor = torch.softmax(logits, dim=1)[0]   # (3,)

    probs = {CLASS_NAMES[i]: float(probs_tensor[i]) for i in range(3)}
    best_idx   = int(torch.argmax(probs_tensor))
    confidence = float(probs_tensor[best_idx])
    label      = CLASS_NAMES[best_idx]
    flagged    = confidence < CONFIDENCE_THRESHOLD

    return label, confidence, probs, flagged
