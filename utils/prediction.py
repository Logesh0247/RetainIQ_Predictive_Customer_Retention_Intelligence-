"""
ML adapter for the integrated RetainIQ UI.
Uses the deployed Logistic Regression model (best F1 on the project test set).
"""
import logging
import numpy as np
from pathlib import Path
from src.preprocessing import preprocess_single

logger = logging.getLogger("retainiq")
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
_MODEL_CACHE = {}

# Held-out test metrics from notebooks/05.Model_training.ipynb
MODEL_COMPARISON = [
    {"name": "Logistic Regression", "accuracy": 0.8034, "f1": 0.6060, "selected": True},
    {"name": "LightGBM", "accuracy": 0.8006, "f1": 0.5957, "selected": False},
    {"name": "XGBoost", "accuracy": 0.7906, "f1": 0.5851, "selected": False},
    {"name": "Random Forest", "accuracy": 0.7899, "f1": 0.5685, "selected": False},
]


class ModelLoadError(RuntimeError):
    pass


def _load(filename):
    import joblib
    path = MODELS_DIR / filename
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {filename}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load model file '{filename}': {exc}") from exc


def load_model(force_reload=False):
    if not force_reload and "best_model" in _MODEL_CACHE:
        return _MODEL_CACHE["best_model"]
    try:
        model = _load("best_model.pkl")
        bundle = {
            "name": "Logistic Regression",
            "model": model,
            "needs_scaling": False,
            "feature_columns": getattr(model, "feature_names_in_", None),
            "metrics": {
                "accuracy": 0.8034,
                "precision": 0.6474,
                "recall": 0.5695,
                "f1": 0.6060,
                "roc_auc": 0.8491,
            },
            "comparison": MODEL_COMPARISON,
        }
        _MODEL_CACHE["best_model"] = bundle
        return bundle
    except ModelLoadError:
        _MODEL_CACHE["best_model"] = None
        return None


def is_model_available():
    return load_model() is not None


def calculate_risk_level(p):
    if p < 0.30:
        return "Low Risk"
    if p < 0.60:
        return "Medium Risk"
    return "High Risk"


def risk_css_class(level):
    return {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(level, "risk-medium")


def signal_strength(p):
    return max(1, min(5, round((1 - p) * 5)))


def _predict_proba(bundle, X):
    return float(bundle["model"].predict_proba(X)[:, 1][0])


def _humanize(name):
    mapping = {
        "Tenure Months": "Account tenure",
        "Monthly Charges": "Monthly charges",
        "Total Charges": "Total charges pattern",
        "Gender_Male": "Male customer",
        "Senior Citizen_Yes": "Senior citizen status",
        "Partner_Yes": "Has a partner",
        "Dependents_Yes": "Has dependents",
        "Phone Service_Yes": "Has phone service",
        "Contract_One year": "One-year contract",
        "Contract_Two year": "Two-year contract",
        "Internet Service_Fiber optic": "Fiber optic internet service",
        "Internet Service_No": "No internet service",
        "Payment Method_Electronic check": "Electronic check payment method",
        "Payment Method_Mailed check": "Mailed check payment method",
        "Payment Method_Credit card (automatic)": "Automatic card payment",
        "Paperless Billing_Yes": "Paperless billing",
        "Tech Support_Yes": "Has tech support",
        "Online Security_Yes": "Has online security",
        "Multiple Lines_Yes": "Multiple phone lines",
    }
    return mapping.get(name, name.replace("_", " "))


def _contributions_from_model(feature_frame):
    """Linear contributions: coefficient * feature value. Honest for LR."""
    bundle = load_model()
    if bundle is None:
        return None
    model = bundle["model"]
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    weights = np.asarray(coef).reshape(-1)
    values = feature_frame.to_numpy(dtype=float)[0]
    if weights.shape[0] != values.shape[0]:
        return None
    return weights * values, feature_frame.columns


def get_top_drivers(feature_frame, top_n=4):
    """Explain this row using LR coefficients (not the old Random Forest SHAP file)."""
    try:
        result = _contributions_from_model(feature_frame)
        if result is None:
            return None
        contribs, names = result
        row = feature_frame.iloc[0]
        numeric = {"Tenure Months", "Monthly Charges", "Total Charges"}
        pairs = []
        for name, value in zip(names, contribs):
            if value <= 0:
                continue
            if name in numeric or row[name] == 1:
                pairs.append((name, float(value)))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [_humanize(name) for name, _ in pairs[:top_n]] or None
    except Exception as exc:
        logger.warning("Explanation failed: %s", exc)
        return None


def predict_customer(record):
    X, cleaned = preprocess_single(record)
    bundle = load_model()
    if bundle is None:
        raise ModelLoadError(
            "The churn prediction model is currently unavailable. "
            "Please make sure models/best_model.pkl exists."
        )
    p = float(np.clip(_predict_proba(bundle, X), 0, 1))
    risk = calculate_risk_level(p)
    return {
        "probability": p,
        "probability_pct": round(p * 100, 1),
        "prediction_label": "Likely to Churn" if p >= 0.5 else "Likely to Stay",
        "will_churn": p >= 0.5,
        "risk_level": risk,
        "risk_css": risk_css_class(risk),
        "signal_bars": signal_strength(p),
        "cleaned_record": cleaned,
        "top_drivers": get_top_drivers(X),
        "model_name": bundle["name"],
        "feature_frame": X,
    }
