import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

from flask import Flask, request, jsonify
from flask_cors import CORS

from qdrant_db import search_similar
from utils import preprocess_image, pil_to_base64
from model import get_model, predict, CLASS_NAMES, get_embedding
from gradcam import generate_heatmap

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AlgoPulse backend running ✅"})


# ─────────────────────────────────────────────
# MAIN PREDICTION ROUTE (DOCTOR DASHBOARD USES THIS)
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_route():

    if "ultrasound" not in request.files:
        return jsonify({"error": "ultrasound image is required"}), 400

    image_bytes = request.files["ultrasound"].read()

    # ── 1. Preprocess image ─────────────────────
    try:
        image_tensor, pil_image, size = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

    # ── 2. Prediction ───────────────────────────
    try:
        label, confidence, probs, flagged = predict(image_tensor)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # ── 3. Grad-CAM ─────────────────────────────
    try:
        model = get_model()
        class_idx = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
        heatmap = generate_heatmap(model, image_tensor, pil_image, class_idx)
    except Exception as e:
        print("⚠️ GradCAM error:", e)
        heatmap = pil_to_base64(pil_image)

    # ── 4. Qdrant Similar Search ────────────────
    similar_cases = []
    try:
        embedding = get_embedding(image_tensor)
        similar_cases = search_similar(embedding, top_k=5)  # Get more results
        # Filter out very high similarity (likely the same image)
        similar_cases = [case for case in similar_cases if case['similarity_score'] < 0.95]
        # Take top 3 after filtering
        similar_cases = similar_cases[:3]
    except Exception as e:
        print("⚠️ Qdrant error:", e)

    # ── 5. Recommendation ────────────────────────
    recommendation_map = {
        "Benign": "Low risk detected. Routine follow-up recommended.",
        "Malignant": "High risk detected. Immediate oncologist consultation advised.",
        "Normal": "No abnormality detected. Continue routine screening."
    }

    recommendation = recommendation_map.get(label, "Consult a doctor.")

    # ── 6. RESPONSE ─────────────────────────────
    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "flagged": flagged,
        "heatmap": heatmap,
        "similar_cases": similar_cases,
        "recommendation": recommendation,
        "image_size": f"{size[0]}x{size[1]}",
        "disclaimer": "AI tool only. Not a final medical diagnosis."
    })


# ─────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting AlgoPulse backend at http://localhost:5000")
    get_model()
app.run(debug=True, use_reloader=False, port=5000)