"""
app.py
------
Flask web application for the Telecom Customer Churn Prediction System.

Routes:
  /             -> Home page with prediction form + dashboard
  /predict      -> POST endpoint (AJAX) that returns churn prediction as JSON
  /dashboard    -> Shows model metrics + EDA plots

Run with:
    python app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.pkl")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

app = Flask(__name__)

# ---------------- Load model artifacts at startup ----------------
model = None
encoders = None
scaler = None
feature_columns = None
metrics_data = None

try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    metrics_data = joblib.load(METRICS_PATH)
    print("✅ Model artifacts loaded successfully.")
except FileNotFoundError:
    print("⚠️  Model artifacts not found. Run clean.py then train.py first.")


# Form field definitions -> shown dynamically on the frontend
FORM_FIELDS = [
    {"name": "gender", "label": "Gender", "type": "select", "options": ["Male", "Female"]},
    {"name": "SeniorCitizen", "label": "Senior Citizen", "type": "select", "options": ["0", "1"]},
    {"name": "Partner", "label": "Has Partner", "type": "select", "options": ["Yes", "No"]},
    {"name": "Dependents", "label": "Has Dependents", "type": "select", "options": ["Yes", "No"]},
    {"name": "tenure", "label": "Tenure (months)", "type": "number"},
    {"name": "PhoneService", "label": "Phone Service", "type": "select", "options": ["Yes", "No"]},
    {"name": "MultipleLines", "label": "Multiple Lines", "type": "select", "options": ["Yes", "No", "No phone service"]},
    {"name": "InternetService", "label": "Internet Service", "type": "select", "options": ["DSL", "Fiber optic", "No"]},
    {"name": "OnlineSecurity", "label": "Online Security", "type": "select", "options": ["Yes", "No", "No internet service"]},
    {"name": "OnlineBackup", "label": "Online Backup", "type": "select", "options": ["Yes", "No", "No internet service"]},
    {"name": "DeviceProtection", "label": "Device Protection", "type": "select", "options": ["Yes", "No", "No internet service"]},
    {"name": "TechSupport", "label": "Tech Support", "type": "select", "options": ["Yes", "No", "No internet service"]},
    {"name": "StreamingTV", "label": "Streaming TV", "type": "select", "options": ["Yes", "No", "No internet service"]},
    {"name": "StreamingMovies", "label": "Streaming Movies", "type": "select", "options": ["Yes", "No", "No internet service"]},
    {"name": "Contract", "label": "Contract Type", "type": "select", "options": ["Month-to-month", "One year", "Two year"]},
    {"name": "PaperlessBilling", "label": "Paperless Billing", "type": "select", "options": ["Yes", "No"]},
    {"name": "PaymentMethod", "label": "Payment Method", "type": "select", "options": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]},
    {"name": "MonthlyCharges", "label": "Monthly Charges ($)", "type": "number"},
    {"name": "TotalCharges", "label": "Total Charges ($)", "type": "number"},
]


def preprocess_input(form_data: dict) -> pd.DataFrame:
    """
    Converts raw form input into the exact encoded/scaled format
    the trained model expects.
    """
    row = {}
    for field in FORM_FIELDS:
        name = field["name"]
        value = form_data.get(name)

        if field["type"] == "number":
            value = float(value)

        row[name] = value

    df = pd.DataFrame([row])

    # Apply the same label encoders used during training
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen categories gracefully
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Ensure SeniorCitizen is numeric (0/1)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Apply the same scaler used during training
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Reorder columns exactly as during training
    df = df[feature_columns]

    return df


@app.route("/")
def home():
    return render_template("index.html", form_fields=FORM_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run clean.py and train.py first."}), 500

    try:
        form_data = request.get_json()
        X = preprocess_input(form_data)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # probability of churn (class 1)

        result = {
            "churn": bool(prediction),
            "churn_label": "Likely to Churn" if prediction == 1 else "Likely to Stay",
            "probability": round(float(probability) * 100, 2),
            "risk_level": (
                "High" if probability >= 0.66 else "Medium" if probability >= 0.33 else "Low"
            ),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/dashboard")
def dashboard():
    plots = []
    plots_dir = os.path.join("static", "plots")
    if os.path.exists(plots_dir):
        plots = sorted(os.listdir(plots_dir))

    return render_template(
        "dashboard.html",
        metrics=metrics_data,
        plots=plots
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
