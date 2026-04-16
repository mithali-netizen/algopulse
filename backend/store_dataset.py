import os
import numpy as np
from pathlib import Path
from utils import preprocess_image
from model import get_embedding
from qdrant_db import create_collection, store_case

# ── Config ─────────────────────────────────────────────
backend_dir = Path(__file__).resolve().parent

# Allow overriding dataset folder from env var.
# Expected structure:
#   <DATASET_PATH>/
#     benign/     malignant/     normal/
DATASET_PATH = os.getenv("DATASET_PATH")
if not DATASET_PATH:
    # Try common folder names next to backend.
    if (backend_dir / "Dataset").exists():
        DATASET_PATH = str(backend_dir / "Dataset")
    elif (backend_dir / "dataset").exists():
        DATASET_PATH = str(backend_dir / "dataset")
    else:
        # Fall back to repo-level `Dataset/` to preserve original behavior.
        DATASET_PATH = "Dataset"

DATASET_PATH = str(DATASET_PATH).rstrip("/\\")

LABELS = ["benign", "malignant", "normal"]
LABELS_TO_CLASSNAMES = {
    "benign": "Benign",
    "malignant": "Malignant",
    "normal": "Normal",
}

# ── Create Qdrant collection ───────────────────────────
create_collection()

# ── Loop through dataset and store embeddings ──────────
for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        try:
            # Read image as bytes
            with open(img_path, 'rb') as f:
                image_bytes = f.read()

            # Preprocess using the same function as app.py
            image_tensor, _, _ = preprocess_image(image_bytes)

            # Get embedding using the same model
            embedding = get_embedding(image_tensor).tolist()

            # Store labels with the same casing as model/app for UI consistency.
            store_case(embedding, LABELS_TO_CLASSNAMES.get(label, label), img_path)
            print(f"✅ Stored: {img_path} → {LABELS_TO_CLASSNAMES.get(label, label)}")

        except Exception as e:
            print(f"⚠️ Failed to process {img_path}: {e}")

        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")