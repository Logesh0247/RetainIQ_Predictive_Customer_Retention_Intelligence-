"""
ML adapter for the integrated RetainIQ UI.
Uses the deployed Random Forest model and feature schema from the second project.
"""
import os, logging, joblib
import numpy as np
from pathlib import Path
from src.preprocessing import preprocess_single

logger = logging.getLogger("retainiq")
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
RISK_THRESHOLDS = {"low_max":0.30, "medium_max":0.60}
_MODEL_CACHE = {}

class ModelLoadError(RuntimeError): pass

def _load(filename):
    path = MODELS_DIR / filename
    if not path.exists(): raise ModelLoadError(f"Model file not found: {filename}")
    try: return joblib.load(path)
    except Exception as exc: raise ModelLoadError(f"Could not load model file '{filename}': {exc}") from exc

def load_model(force_reload=False):
    if not force_reload and "best_model" in _MODEL_CACHE: return _MODEL_CACHE["best_model"]
    try:
        model = _load("best_model.pkl")
        bundle = {
            "name":"Random Forest",
            "model":model,
            "needs_scaling":False,
            "feature_columns": getattr(model, "feature_names_in_", None),
            "metrics": {},
        }
        _MODEL_CACHE["best_model"] = bundle
        return bundle
    except ModelLoadError:
        _MODEL_CACHE["best_model"] = None
        return None

def load_shap_explainer():
    if "shap_explainer" not in _MODEL_CACHE:
        try: _MODEL_CACHE["shap_explainer"] = _load("shap_explainer.pkl")
        except Exception as exc:
            logger.warning("SHAP explainer unavailable: %s", exc); _MODEL_CACHE["shap_explainer"] = None
    return _MODEL_CACHE["shap_explainer"]

def is_model_available(): return load_model() is not None

def calculate_risk_level(p):
    if p < .30: return "Low Risk"
    if p < .60: return "Medium Risk"
    return "High Risk"

def risk_css_class(level):
    return {"Low Risk":"risk-low","Medium Risk":"risk-medium","High Risk":"risk-high"}.get(level,"risk-medium")

def signal_strength(p):
    return max(1,min(5,round((1-p)*5)))

def _predict_proba(bundle, X):
    return float(bundle["model"].predict_proba(X)[:,1][0])

def _humanize(name):
    mapping = {
        "Tenure Months":"Short account tenure","Monthly Charges":"High monthly charges",
        "Total Charges":"Total charges pattern","Contract_One year":"One-year contract",
        "Contract_Two year":"Two-year contract","Internet Service_Fiber optic":"Fiber optic internet service",
        "Internet Service_No":"No internet service","Payment Method_Electronic check":"Electronic check payment method",
        "Paperless Billing_Yes":"Paperless billing","Tech Support_Yes":"Has tech support",
        "Online Security_Yes":"Has online security","Senior Citizen_Yes":"Senior citizen status",
    }
    return mapping.get(name,name.replace("_"," "))

def get_top_drivers(feature_frame, top_n=4):
    explainer=load_shap_explainer()
    if explainer is None: return None
    try:
        sv=explainer.shap_values(feature_frame)
        if isinstance(sv,list): values=sv[1][0]
        elif getattr(sv,"ndim",0)==3: values=sv[0,:,1]
        else: values=sv[0]
        row=feature_frame.iloc[0]
        numeric={"Tenure Months","Monthly Charges","Total Charges","Senior Citizen_Yes"}
        pairs=[(n,v) for n,v in zip(feature_frame.columns,values)
               if v>0 and (n in numeric or row[n]==1)]
        pairs.sort(key=lambda x:x[1],reverse=True)
        return [_humanize(n) for n,_ in pairs[:top_n]] or None
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc); return None

def predict_customer(record):
    X, cleaned = preprocess_single(record)
    bundle=load_model()
    if bundle is None: raise ModelLoadError("The churn prediction model is currently unavailable. Please make sure models/best_model.pkl exists.")
    p=float(np.clip(_predict_proba(bundle,X),0,1))
    risk=calculate_risk_level(p)
    return {
        "probability":p,"probability_pct":round(p*100,1),
        "prediction_label":"Likely to Churn" if p>=.5 else "Likely to Stay",
        "will_churn":p>=.5,"risk_level":risk,"risk_css":risk_css_class(risk),
        "signal_bars":signal_strength(p),"cleaned_record":cleaned,
        "top_drivers":get_top_drivers(X),"model_name":bundle["name"],
    }
