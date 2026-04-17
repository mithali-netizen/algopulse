import os
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, url_for
from flask_cors import CORS

from gradcam import generate_heatmap
from model import CLASS_NAMES, get_embedding, get_model, predict
from qdrant_db import search_similar
from utils import pil_to_base64, preprocess_image

app = Flask(__name__)
CORS(app)


def get_allowed_case_roots():
    backend_dir = Path(__file__).resolve().parent
    repo_dir = backend_dir.parent
    configured = os.getenv("DATASET_PATH")

    roots = []
    if configured:
        roots.append(Path(configured))

    roots.extend([
        backend_dir / "Dataset",
        backend_dir / "dataset",
        repo_dir / "Dataset",
        repo_dir / "dataset",
    ])

    resolved_roots = []
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if resolved.exists() and resolved not in resolved_roots:
            resolved_roots.append(resolved)
    return resolved_roots


def resolve_case_image_path(raw_path):
    if not raw_path:
        return None

    requested = Path(raw_path)
    backend_dir = Path(__file__).resolve().parent
    repo_dir = backend_dir.parent

    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend([
            Path.cwd() / requested,
            backend_dir / requested,
            repo_dir / requested,
        ])

    allowed_roots = get_allowed_case_roots()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        if any(root == resolved or root in resolved.parents for root in allowed_roots):
            return resolved

    # Fallback for older Qdrant entries whose stored folder path no longer matches
    # the current dataset location. We search by filename inside approved dataset roots.
    requested_name = requested.name
    if requested_name:
        for root in allowed_roots:
            matches = list(root.rglob(requested_name))
            if matches:
                return matches[0]
    return None


def attach_case_image_urls(cases):
    enriched_cases = []
    for case in cases:
        image_path = case.get("image_path", "")
        resolved_path = resolve_case_image_path(image_path)
        image_name = Path(image_path).name if image_path else ""
        enriched_case = dict(case)
        enriched_case["image_name"] = image_name
        enriched_case["image_url"] = (
            url_for("case_image", name=image_name, _external=True)
            if image_name else None
        )
        enriched_cases.append(enriched_case)
    return enriched_cases


def dedupe_similar_cases(cases, limit=3):
    unique_cases = []
    seen_names = set()

    for case in cases:
        image_path = case.get("image_path", "")
        image_name = Path(image_path).name.lower() if image_path else ""
        if not image_name or image_name in seen_names:
            continue
        seen_names.add(image_name)
        unique_cases.append(case)
        if len(unique_cases) >= limit:
            break

    return unique_cases


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AlgoPulse backend running"})


@app.route("/case-image", methods=["GET"])
def case_image():
    image_path = request.args.get("path", "")
    image_name = request.args.get("name", "")
    resolved_path = resolve_case_image_path(image_path or image_name)

    if resolved_path is None:
        abort(404, description="Case image not found")

    return send_file(resolved_path)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "ultrasound" not in request.files:
        return jsonify({"error": "ultrasound image is required"}), 400

    image_bytes = request.files["ultrasound"].read()

    try:
        image_tensor, pil_image, size = preprocess_image(image_bytes)
    except Exception as exc:
        return jsonify({"error": f"Preprocessing failed: {exc}"}), 400

    try:
        label, confidence, probs, flagged = predict(image_tensor)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    try:
        model = get_model()
        class_idx = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
        heatmap = generate_heatmap(model, image_tensor, pil_image, class_idx)
    except Exception as exc:
        print("GradCAM error:", exc)
        heatmap = pil_to_base64(pil_image)

    similar_cases = []
    try:
        embedding = get_embedding(image_tensor)
        similar_cases = search_similar(embedding, top_k=12)
        similar_cases = [case for case in similar_cases if case["similarity_score"] < 0.95]
        similar_cases = dedupe_similar_cases(similar_cases, limit=3)
        similar_cases = attach_case_image_urls(similar_cases)
    except Exception as exc:
        print("Qdrant error:", exc)

    recommendation_map = {
        "Benign": "Low risk detected. Routine follow-up recommended.",
        "Malignant": "High risk detected. Immediate oncologist consultation advised.",
        "Normal": "No abnormality detected. Continue routine screening.",
    }
    recommendation = recommendation_map.get(label, "Consult a doctor.")

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "flagged": flagged,
        "heatmap": heatmap,
        "similar_cases": similar_cases,
        "recommendation": recommendation,
        "image_size": f"{size[0]}x{size[1]}",
        "disclaimer": "AI tool only. Not a final medical diagnosis.",
    })


if __name__ == "__main__":
    print("Starting AlgoPulse backend at http://localhost:5000")
    get_model()
    app.run(debug=True, use_reloader=False, port=5000)
