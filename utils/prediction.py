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


# Single source of truth for the risk bands. Every scoring path (single,
# bulk Telco, universal engine) imports this so the same customer never gets
# two different risk labels in two different screens.
MEDIUM_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def calculate_risk_level(p):
    if p < MEDIUM_RISK_THRESHOLD:
        return "Low Risk"
    if p < HIGH_RISK_THRESHOLD:
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


# ---------------------------------------------------------------------------
# Signed explanation: what pushes this customer toward churn, and what holds
# them back. Logistic regression makes this exact rather than approximate.
# ---------------------------------------------------------------------------

def explain_prediction(feature_frame, top_n=6):
    """Return signed, ranked contributions for one scored customer.

    Each item: {label, contribution, direction, share} where `share` is the
    contribution as a percentage of the largest absolute contribution, so the
    UI can draw comparable bars.
    """
    try:
        result = _contributions_from_model(feature_frame)
        if result is None:
            return []
        contribs, names = result
        row = feature_frame.iloc[0]
        numeric = {"Tenure Months", "Monthly Charges", "Total Charges"}
        items = []
        for name, value in zip(names, contribs):
            # Skip one-hot features the customer does not have: a zero-valued
            # dummy contributes nothing and would only add noise.
            if name not in numeric and row[name] != 1:
                continue
            contribution = float(value)
            if abs(contribution) < 1e-9:
                continue
            items.append({
                "label": _humanize(name),
                "feature": name,
                "contribution": contribution,
                "direction": "risk" if contribution > 0 else "protective",
            })
        if not items:
            return []
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        items = items[:top_n]
        strongest = max(abs(item["contribution"]) for item in items) or 1.0
        for item in items:
            item["share"] = round(abs(item["contribution"]) / strongest * 100, 1)
        return items
    except Exception as exc:
        logger.warning("Signed explanation failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# What-if simulator: re-score the same customer under retention levers a team
# could actually pull, so the page answers "what should we do?" and not just
# "how bad is it?".
# ---------------------------------------------------------------------------

def _record_value(record, key, default=""):
    try:
        value = record.get(key, default)
    except AttributeError:
        value = default
    if value is None:
        return default
    return value


def _lever_definitions(record):
    """Levers that make sense for this specific customer, in business order."""
    contract = str(_record_value(record, "Contract", "Month-to-month"))
    payment = str(_record_value(record, "PaymentMethod", ""))
    internet = str(_record_value(record, "InternetService", "No"))
    has_internet = internet.strip().lower() != "no"
    try:
        monthly = float(_record_value(record, "MonthlyCharges", 0) or 0)
    except (TypeError, ValueError):
        monthly = 0.0
    try:
        tenure = int(float(_record_value(record, "tenure", 0) or 0))
    except (TypeError, ValueError):
        tenure = 0

    levers = []

    if contract != "One year":
        levers.append({
            "id": "contract_one_year",
            "label": "Move to a one-year contract",
            "detail": "Offer a discounted rate in exchange for a 12-month commitment.",
            "changes": {"Contract": "One year"},
        })
    if contract != "Two year":
        levers.append({
            "id": "contract_two_year",
            "label": "Move to a two-year contract",
            "detail": "The strongest retention lever in the model — best paired with an incentive.",
            "changes": {"Contract": "Two year"},
        })
    if payment in ("Electronic check", "Mailed check"):
        levers.append({
            "id": "auto_pay",
            "label": "Switch to automatic bank payment",
            "detail": "Manual payment methods correlate strongly with churn; auto-pay removes the monthly friction.",
            "changes": {"PaymentMethod": "Bank transfer (automatic)"},
        })
    if has_internet and str(_record_value(record, "TechSupport", "")) != "Yes":
        levers.append({
            "id": "tech_support",
            "label": "Add Tech Support",
            "detail": "Offer a free trial period, then a bundled rate.",
            "changes": {"TechSupport": "Yes"},
        })
    if has_internet and str(_record_value(record, "OnlineSecurity", "")) != "Yes":
        levers.append({
            "id": "online_security",
            "label": "Add Online Security",
            "detail": "A low-cost add-on that measurably increases stickiness.",
            "changes": {"OnlineSecurity": "Yes"},
        })
    if has_internet and str(_record_value(record, "OnlineBackup", "")) != "Yes":
        levers.append({
            "id": "online_backup",
            "label": "Add Online Backup",
            "detail": "Bundle at a promotional rate alongside the existing plan.",
            "changes": {"OnlineBackup": "Yes"},
        })
    if monthly > 0:
        discounted = round(monthly * 0.90, 2)
        levers.append({
            "id": "discount_10",
            "label": "Apply a 10% loyalty discount",
            "detail": f"Monthly charges ${monthly:,.2f} → ${discounted:,.2f}.",
            "changes": {"MonthlyCharges": discounted},
        })
    if internet == "Fiber optic":
        levers.append({
            "id": "fiber_to_dsl",
            "label": "Right-size fiber to DSL",
            "detail": "Fiber customers churn more in this dataset — worth testing where speed is not critical.",
            "changes": {"InternetService": "DSL"},
        })
    levers.append({
        "id": "retain_12_months",
        "label": "Projection: keep them 12 more months",
        "detail": "Not an action — shows how the risk profile matures if retention succeeds.",
        "changes": {
            "tenure": min(tenure + 12, 100),
            "TotalCharges": round(
                float(_record_value(record, "TotalCharges", 0) or 0) + monthly * 12, 2
            ),
        },
        "projection": True,
    })
    return levers


def simulate_what_if(record, base_probability=None, top_n=6):
    """Re-score `record` under each retention lever.

    Returns a list sorted by impact (largest risk reduction first). Each item
    carries the new probability, the delta in percentage points, and whether
    the change helps or hurts.
    """
    bundle = load_model()
    if bundle is None:
        return []
    try:
        base_record = dict(record)
    except (TypeError, ValueError):
        return []

    if base_probability is None:
        try:
            X, _ = preprocess_single(base_record)
            base_probability = float(np.clip(_predict_proba(bundle, X), 0, 1))
        except Exception as exc:
            logger.warning("What-if baseline scoring failed: %s", exc)
            return []

    scenarios = []
    for lever in _lever_definitions(base_record):
        candidate = dict(base_record)
        candidate.update(lever["changes"])
        try:
            X, _ = preprocess_single(candidate)
            probability = float(np.clip(_predict_proba(bundle, X), 0, 1))
        except Exception as exc:
            logger.warning("What-if scenario '%s' failed: %s", lever["id"], exc)
            continue
        delta = probability - base_probability
        risk = calculate_risk_level(probability)
        scenarios.append({
            "id": lever["id"],
            "label": lever["label"],
            "detail": lever["detail"],
            "projection": lever.get("projection", False),
            "probability": probability,
            "probability_pct": round(probability * 100, 1),
            "delta_points": round(delta * 100, 1),
            "improves": delta < -0.001,
            "worsens": delta > 0.001,
            "risk_level": risk,
            "risk_css": risk_css_class(risk),
            "changes": lever["changes"],
        })

    scenarios.sort(key=lambda item: item["delta_points"])
    return scenarios[:top_n]
