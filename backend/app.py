from flask import Flask, request, jsonify
from flask_cors import CORS

from utils import preprocess_image, pil_to_base64
from model import get_model, predict, CLASS_NAMES
from gradcam import generate_heatmap

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AlgoPulse backend is running ✅"})


@app.route("/predict", methods=["POST"])
def predict_route():
    if "ultrasound" not in request.files:
        return jsonify({"error": "ultrasound image is required"}), 400

    image_bytes = request.files["ultrasound"].read()

    # ── Preprocess — returns tensor, pil_image, and size ──────────────────
    try:
        image_tensor, pil_image, size = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 400

    # ── Predict ────────────────────────────────────────────────────────────
    try:
        label, confidence, probs, flagged = predict(image_tensor)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    # ── Grad-CAM — heatmap matches original image size ─────────────────────
    try:
        model     = get_model()
        class_idx = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
        heatmap   = generate_heatmap(model, image_tensor, pil_image, class_idx)
    except Exception as e:
        print(f"⚠️  Heatmap error: {e}")
        heatmap = pil_to_base64(pil_image)

    # ── Recommendation ─────────────────────────────────────────────────────
    recommendation = {
        "Benign"   : "Low risk detected. Routine follow-up in 6–12 months recommended.",
        "Malignant": "High risk detected. Immediate referral to oncologist for biopsy advised.",
        "Normal"   : "No abnormality detected. Continue routine screening as advised.",
    }.get(label, "Please consult a clinician.")

    flag_message = "⚠️ Low confidence — clinical review strongly recommended." if flagged else None

    return jsonify({
        "label"         : label,
        "confidence"    : round(confidence, 4),
        "probabilities" : {k: round(v, 4) for k, v in probs.items()},
        "flagged"       : flagged,
        "flag_message"  : flag_message,
        "heatmap"       : heatmap,
        "recommendation": recommendation,
        "image_size"    : f"{size[0]}x{size[1]}",
        "metrics_info"  : "Evaluated on accuracy, precision, recall, F1-score.",
        "disclaimer"    : "This is an AI decision-support tool. Final diagnosis must be made by a qualified clinician."
    })


if __name__ == "__main__":
    print("🚀 Starting AlgoPulse backend on http://localhost:5000")
    get_model()
    app.run(debug=True, port=5000)