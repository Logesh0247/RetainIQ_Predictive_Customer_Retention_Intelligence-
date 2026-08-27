"""
Optional FastAPI wrapper around the same prediction engine as the Flask UI.
Run: uvicorn api.app:app --host 0.0.0.0 --port 8000
The Flask app also exposes /api/health and /api/predict.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import PreprocessingError
from utils.prediction import predict_customer, is_model_available, load_model, ModelLoadError
from utils.recommendation import generate_recommendation


class CustomerRequest(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None


app = FastAPI(
    title="RetainIQ Churn Prediction API",
    description="Same Logistic Regression model used by the Signal Ops Console.",
    version="1.0.0",
)


@app.get("/")
@app.get("/health")
def health():
    bundle = load_model()
    return {
        "status": "Running",
        "model_ready": is_model_available(),
        "model": bundle.get("name") if bundle else None,
    }


@app.post("/predict")
def predict(customer: CustomerRequest):
    try:
        result = predict_customer(customer.model_dump())
        recs = generate_recommendation(
            result["cleaned_record"],
            result["probability"],
            result["risk_level"],
        )
        return {
            "customer_id": str(result["cleaned_record"].get("customerID", "")),
            "probability": result["probability"],
            "probability_pct": result["probability_pct"],
            "prediction": result["prediction_label"],
            "risk_level": result["risk_level"],
            "top_drivers": result["top_drivers"],
            "primary_action": recs[0] if recs else "Monitor account.",
            "recommendations": recs,
            "model_name": result["model_name"],
        }
    except (PreprocessingError, ModelLoadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
