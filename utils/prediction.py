from pathlib import Path

import joblib

from src.preprocessing import preprocess_input
from src.recommendation import get_risk, get_recommendation

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_customer(form):
    input_df = preprocess_input(form)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    risk = get_risk(probability)

    customer = {

        "Contract": form["Contract"],

        "Monthly Charges": float(form["Monthly Charges"]),

        "Tenure Months": float(form["Tenure Months"]),

        "Internet Service": form["Internet Service"],

        "Online Security": form["Online Security"],

        "Online Backup": form["Online Backup"],

        "Device Protection": form["Device Protection"],

        "Tech Support": form["Tech Support"],

        "Streaming TV": form["Streaming TV"],

        "Streaming Movies": form["Streaming Movies"],

        "Multiple Lines": form["Multiple Lines"],

        "Payment Method": form["Payment Method"],

        "Paperless Billing": form["Paperless Billing"],

        "Partner": form["Partner"],

        "Dependents": form["Dependents"],

        "Senior Citizen": form["Senior Citizen"],

        "Phone Service": form["Phone Service"]

    }

    recommendation = get_recommendation(customer, risk)

    result = "Customer Will Churn"
    if prediction == 0:
        result = "Customer Will Stay"

    return {
        "prediction": result,
        "probability": round(probability * 100, 2),
        "risk": risk,
        "recommendation": recommendation,
    }