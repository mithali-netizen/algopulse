import os
import numpy as np
from utils import preprocess_image
from model import get_embedding
from qdrant_db import create_collection, store_case

# ── Config ─────────────────────────────────────────────
DATASET_PATH = "Dataset/"   # Your dataset folder (note: capital D)
LABELS = ["benign", "malignant", "normal"]

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

            store_case(embedding, label, img_path)
            print(f"✅ Stored: {img_path} → {label}")

        except Exception as e:
            print(f"⚠️ Failed to process {img_path}: {e}")

        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")