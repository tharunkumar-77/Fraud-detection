"""
Flask API + frontend for the fraud detection model.

Prerequisite: run train_fraud_model.py first so model/*.pkl exists.

Usage:
    python app.py
Then open http://127.0.0.1:5001
"""

import os
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "best_threshold.pkl")

model = None
feature_columns = None
threshold = 0.5

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    if os.path.exists(THRESHOLD_PATH):
        threshold = joblib.load(THRESHOLD_PATH)
    print(f"Model loaded. {len(feature_columns)} features. Threshold={threshold}")
else:
    print("WARNING: model/fraud_model.pkl not found. Run train_fraud_model.py first.")


@app.route("/")
def home():
    columns = feature_columns if feature_columns else []
    return render_template("index.html", columns=columns, threshold=threshold)


@app.route("/api/health")
def health():
    return jsonify({
        "model_loaded": model is not None,
        "n_features": len(feature_columns) if feature_columns else 0,
        "threshold": threshold
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_fraud_model.py first."}), 500

    data = request.get_json(force=True)

    try:
        row = [float(data.get(col, 0)) for col in feature_columns]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    X = np.array(row).reshape(1, -1)
    fraud_prob = model.predict_proba(X)[0, 1]
    is_fraud = bool(fraud_prob >= threshold)

    return jsonify({
        "fraud_probability": round(float(fraud_prob), 6),
        "threshold_used": threshold,
        "is_fraud": is_fraud,
        "verdict": "ANOMALY / LIKELY FRAUD" if is_fraud else "LEGITIMATE"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
