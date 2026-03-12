from flask import Flask, request, jsonify
from flask_cors import CORS

from utils import preprocess_image, pil_to_base64
from model import get_model, predict, CLASS_NAMES
from gradcam import generate_heatmap

app = Flask(__name__)
CORS(app)   # allows React (localhost:3000) to talk to Flask (localhost:5000)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AlgoPulse backend is running ✅"})


# ── Main prediction route ─────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Expects a multipart form with:
        ultrasound  (required) — the uploaded ultrasound image

    Returns JSON:
    {
        "label"          : "Malignant",
        "confidence"     : 0.91,
        "probabilities"  : {"Benign": 0.05, "Malignant": 0.91, "Normal": 0.04},
        "flagged"        : false,
        "heatmap"        : "data:image/png;base64,...",
        "recommendation" : "High risk detected...",
        "metrics_info"   : "Evaluated on accuracy, precision, recall, F1-score",
        "disclaimer"     : "..."
    }
    """

    # ── 1. Read uploaded file ───────────────────────────────────────────────
    if "ultrasound" not in request.files:
        return jsonify({"error": "ultrasound image is required"}), 400

    image_bytes = request.files["ultrasound"].read()

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    try:
        image_tensor, pil_image = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 400

    # ── 3. Run inference ────────────────────────────────────────────────────
    try:
        label, confidence, probs, flagged = predict(image_tensor)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    # ── 4. Generate Grad-CAM heatmap ────────────────────────────────────────
    try:
        model     = get_model()
        class_idx = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
        heatmap   = generate_heatmap(model, image_tensor, pil_image, class_idx)
    except Exception as e:
        print(f"⚠️  Heatmap error: {e}")
        heatmap = pil_to_base64(pil_image)

    # ── 5. Recommendation text ──────────────────────────────────────────────
    recommendation = {
        "Benign"   : "Low risk detected. Routine follow-up in 6–12 months recommended.",
        "Malignant": "High risk detected. Immediate referral to oncologist for biopsy advised.",
        "Normal"   : "No abnormality detected. Continue routine screening as advised.",
    }.get(label, "Please consult a clinician.")

    # Flag message if confidence is low
    flag_message = "⚠️ Low confidence — clinical review strongly recommended." if flagged else None

    # ── 6. Return JSON response ─────────────────────────────────────────────
    return jsonify({
        "label"         : label,
        "confidence"    : round(confidence, 4),
        "probabilities" : {k: round(v, 4) for k, v in probs.items()},
        "flagged"       : flagged,
        "flag_message"  : flag_message,
        "heatmap"       : heatmap,
        "recommendation": recommendation,
        "metrics_info"  : "Model evaluated on accuracy, precision, recall, F1-score and confusion matrix.",
        "disclaimer"    : "This is an AI decision-support tool. Final diagnosis must be made by a qualified clinician."
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting AlgoPulse backend on http://localhost:5000")
    get_model()    # preload model at startup
    app.run(debug=True, port=5000)
