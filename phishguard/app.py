import sys
import os
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# ==============================
# Path Configuration
# ==============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from utils.feature_extraction import extract_features

# ==============================
# Flask App Initialization
# ==============================
app = Flask(__name__)

# ==============================
# Load Trained Model Safely
# ==============================
model_path = os.path.join(BASE_DIR, "models", "phishguard_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found in /models folder.")

model = joblib.load(model_path)


# ==============================
# Home Route
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Web Prediction Route
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form.get("url")

        if not url:
            return render_template(
                "result.html",
                result="⚠ Please enter a valid URL",
                probability=0,
                url="",
                bar_color="#999999"
            )

        # Extract features
        features = extract_features(url)
        features = np.array(features).reshape(1, -1)

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][prediction] * 100)

        if prediction == 1:
            result = "🚨 Phishing Website Detected"
            bar_color = "#ff1744"  # Red
        else:
            result = "✅ Legitimate Website"
            bar_color = "#00c853"  # Green

        # ==============================
        # Logging Prediction
        # ==============================
        log_file = os.path.join(BASE_DIR, "prediction_logs.csv")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()},{url},{prediction},{round(probability,2)}\n")

        return render_template(
            "result.html",
            result=result,
            probability=round(probability, 2),
            url=url,
            bar_color=bar_color
        )

    except Exception:
        return render_template(
            "result.html",
            result="❌ Internal Server Error",
            probability=0,
            url="",
            bar_color="#999999"
        )


# ==============================
# API Endpoint (Future Mobile Integration)
# ==============================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        url = data.get("url")

        features = extract_features(url)
        features = np.array(features).reshape(1, -1)

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][prediction] * 100)

        return jsonify({
            "url": url,
            "prediction": prediction,
            "probability": round(probability, 2),
            "status": "Phishing" if prediction == 1 else "Legitimate"
        })

    except Exception:
        return jsonify({"error": "Invalid request"}), 400


# ==============================
# Health Check Route (Deployment Important)
# ==============================
@app.route("/health")
def health():
    return jsonify({"status": "running"})


# ==============================
# Run Application
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
