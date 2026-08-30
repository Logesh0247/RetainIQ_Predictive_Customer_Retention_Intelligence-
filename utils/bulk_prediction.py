"""
Bulk scoring adapter for the RetainIQ UI.
Uses the integrated project's Logistic Regression model and canonical preprocessing.
"""
import os
import io
import pickle
import logging
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_bulk, PreprocessingError
from src.paths import SAMPLE_CUSTOMERS, REPORTS_DIR as PATHS_REPORTS
from utils.prediction import (
    load_model,
    calculate_risk_level,
    risk_css_class,
    signal_strength,
    get_top_drivers,
)
from utils.recommendation import (
    generate_recommendation,
    generate_recommendation_summary,
    retention_action_label,
)

logger = logging.getLogger("retainiq")
ALLOWED_EXTENSIONS = {"csv"}
MAX_ROWS = 20000
TOP_N_WITH_DRIVERS = 10
MAX_DISPLAY_ROWS = 150
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = str(PATHS_REPORTS)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


class BulkPredictionError(ValueError):
    pass


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _predict_batch(bundle, X):
    return bundle["model"].predict_proba(X)[:, 1]


def _score_dataframe(df):
    if df.empty:
        raise BulkPredictionError("The uploaded CSV contains no customer rows.")
    if len(df) > MAX_ROWS:
        raise BulkPredictionError(f"The uploaded CSV has too many rows (max {MAX_ROWS:,}).")
    try:
        X, cleaned = preprocess_bulk(df)
    except PreprocessingError as exc:
        raise BulkPredictionError(str(exc)) from exc
    bundle = load_model()
    if bundle is None:
        raise BulkPredictionError(
            "The churn prediction model is currently unavailable. "
            "Please make sure models/best_model.pkl exists."
        )
    try:
        probs = np.clip(_predict_batch(bundle, X), 0, 1)
    except Exception as exc:
        logger.exception("Bulk model prediction failed")
        raise BulkPredictionError(
            f"An error occurred while running predictions on this dataset: {exc}"
        ) from exc

    cleaned = cleaned.reset_index(drop=True)
    X = X.reset_index(drop=True)
    cleaned_records = cleaned.to_dict("records")

    risk_levels = [calculate_risk_level(p) for p in probs]
    actions = [
        retention_action_label(cleaned_records[i], probs[i], risk_levels[i])
        for i in range(len(cleaned_records))
    ]
    summaries = [
        generate_recommendation_summary(cleaned_records[i], probs[i], risk_levels[i])
        for i in range(len(cleaned_records))
    ]
    at_risk = [level in ("High Risk", "Medium Risk") for level in risk_levels]

    results = pd.DataFrame({
        "Customer ID": cleaned["customerID"].values,
        "Churn Probability": np.round(probs * 100, 1),
        "Churn_Probability": np.round(probs, 4),
        "Predicted_Churn": (probs >= 0.5).astype(int),
        "Prediction": np.where(probs >= 0.5, "Likely to Churn", "Likely to Stay"),
        "Risk Level": risk_levels,
        "Risk_Segment": risk_levels,
        "Monthly Charges": cleaned["MonthlyCharges"].round(2).values,
        "Tenure": cleaned["tenure"].values,
        "Contract": cleaned["Contract"].values,
        "Internet Service": cleaned["InternetService"].values,
        "Payment Method": cleaned["PaymentMethod"].values,
        "Gender": cleaned["gender"].values,
        "Senior Citizen": cleaned["SeniorCitizen"].values,
        "Recommendation": summaries,
        "Retention_Recommendation": actions,
        "Potential_Revenue_Saved": np.where(at_risk, cleaned["MonthlyCharges"].round(2), 0.0),
    })
    return {
        "results_df": results,
        "cleaned_df": cleaned,
        "feature_frame": X,
        "probabilities": probs,
    }


def run_bulk_prediction_from_bytes(raw, filename="upload.csv"):
    if not raw:
        raise BulkPredictionError("The uploaded file is empty.")
    if filename and not allowed_file(filename):
        raise BulkPredictionError("Invalid file type. Only .csv files are supported.")
    
    df = None
    last_err = None
    for encoding in ["utf-8-sig", "utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except pd.errors.EmptyDataError as exc:
            raise BulkPredictionError("The uploaded CSV file has no data.") from exc
        except Exception as exc:
            last_err = exc
            continue

    if df is None:
        raise BulkPredictionError(
            f"The uploaded file could not be parsed as a valid CSV: {last_err or 'unknown encoding error'}"
        )
    return _score_dataframe(df)


def run_bulk_prediction_from_path(path):
    path = str(path)
    if not os.path.exists(path):
        raise BulkPredictionError("Sample dataset was not found.")
    with open(path, "rb") as handle:
        return run_bulk_prediction_from_bytes(handle.read(), os.path.basename(path))


def run_sample_prediction():
    return run_bulk_prediction_from_path(SAMPLE_CUSTOMERS)


def run_bulk_prediction(file_storage):
    if file_storage is None or getattr(file_storage, "filename", "") == "":
        raise BulkPredictionError("No file was selected. Please choose a CSV file to upload.")
    if not allowed_file(file_storage.filename):
        raise BulkPredictionError("Invalid file type. Only .csv files are supported.")
    try:
        raw = file_storage.read()
    except Exception as exc:
        raise BulkPredictionError(f"Could not read the uploaded file: {exc}") from exc
    return run_bulk_prediction_from_bytes(raw, file_storage.filename)


def _record_from_row(row, cleaned_row, prob, extra=None):
    risk = row["Risk Level"] if hasattr(row, "get") else row.get("Risk Level", "Low Risk") if isinstance(row, dict) else row["Risk Level"]
    m_charges = row["Monthly Charges"] if hasattr(row, "get") else row.get("Monthly Charges", 0) if isinstance(row, dict) else row["Monthly Charges"]
    c_id = row["Customer ID"] if hasattr(row, "get") else row.get("Customer ID", "") if isinstance(row, dict) else row["Customer ID"]
    pred = row["Prediction"] if hasattr(row, "get") else row.get("Prediction", "Likely to Stay") if isinstance(row, dict) else row["Prediction"]
    prob_pct = row["Churn Probability"] if hasattr(row, "get") else row.get("Churn Probability", 0) if isinstance(row, dict) else row["Churn Probability"]
    tenure_val = row["Tenure"] if hasattr(row, "get") else row.get("Tenure", 0) if isinstance(row, dict) else row["Tenure"]
    contract_val = row["Contract"] if hasattr(row, "get") else row.get("Contract", "Month-to-month") if isinstance(row, dict) else row["Contract"]

    recs = generate_recommendation(cleaned_row, prob, risk)
    if not recs:
        recs = ["Monitor account."]

    record = {
        "id": str(c_id),
        "probability_pct": float(prob_pct),
        "prediction": str(pred),
        "will_churn": str(pred) == "Likely to Churn",
        "risk_level": str(risk),
        "risk_css": risk_css_class(str(risk)),
        "signal_bars": signal_strength(float(prob)),
        "monthly_charges": float(m_charges),
        "tenure": int(tenure_val),
        "contract": str(contract_val),
        "recommendations": recs,
        "primary_action": recs[0],
        "is_high": str(risk) == "High Risk",
        "is_mtm": str(contract_val) == "Month-to-month",
        "is_high_bill": float(m_charges) >= 80,
        "row_classes": " ".join(filter(None, [
            "is-high" if str(risk) == "High Risk" else "",
            "is-mtm" if str(contract_val) == "Month-to-month" else "",
            "is-high-bill" if float(m_charges) >= 80 else "",
        ])),
    }
    if extra:
        record.update(extra)
    return record


def build_display_records(bundle, sort_by="probability", max_rows=MAX_DISPLAY_ROWS):
    results, cleaned, probs = bundle["results_df"], bundle["cleaned_df"], bundle["probabilities"]
    n_rows = len(results)
    
    if sort_by == "charges":
        order = np.argsort(-results["Monthly Charges"].values)
    elif sort_by == "tenure":
        order = np.argsort(results["Tenure"].values)
    else:  # probability
        order = np.argsort(-probs)

    top_indices = order[:max_rows]
    records = []
    for idx in top_indices:
        idx = int(idx)
        records.append(_record_from_row(results.iloc[idx], cleaned.iloc[idx], float(probs[idx])))
    return records


def build_top_risk_records(bundle, top_n=TOP_N_WITH_DRIVERS):
    results, cleaned, features, probs = (
        bundle["results_df"],
        bundle["cleaned_df"],
        bundle["feature_frame"],
        bundle["probabilities"],
    )
    records = []
    top_indices = np.argsort(-probs)[:top_n]
    for idx in top_indices:
        idx = int(idx)
        risk = str(results.iloc[idx]["Risk Level"])
        recs = generate_recommendation(cleaned.iloc[idx], float(probs[idx]), risk)
        if not recs:
            recs = ["Monitor account."]
        records.append(_record_from_row(
            results.iloc[idx],
            cleaned.iloc[idx],
            float(probs[idx]),
            extra={
                "internet_service": cleaned.iloc[idx].get("InternetService"),
                "payment_method": cleaned.iloc[idx].get("PaymentMethod"),
                "top_drivers": get_top_drivers(features.iloc[[idx]]),
                "recommendations": recs,
                "primary_action": recs[0],
            },
        ))
    return records


def compute_dashboard_kpis(results_df):
    if results_df is None or results_df.empty:
        return None
    total = len(results_df)
    churners = int((results_df["Prediction"] == "Likely to Churn").sum())
    high = int((results_df["Risk Level"] == "High Risk").sum())
    med = int((results_df["Risk Level"] == "Medium Risk").sum())
    low = int((results_df["Risk Level"] == "Low Risk").sum())
    pct = lambda n: round(n / total * 100, 1) if total else 0.0
    high_rev = float(results_df.loc[results_df["Risk Level"] == "High Risk", "Monthly Charges"].sum())
    at_risk_rev = float(
        results_df.loc[results_df["Risk Level"].isin(["High Risk", "Medium Risk"]), "Monthly Charges"].sum()
    )
    return {
        "total_customers": total,
        "predicted_churners": churners,
        "churn_rate": round(churners / total * 100, 1),
        "avg_churn_probability": round(results_df["Churn Probability"].mean(), 1),
        "high_risk_count": high,
        "medium_risk_count": med,
        "low_risk_count": low,
        "high_risk_pct": pct(high),
        "medium_risk_pct": pct(med),
        "low_risk_pct": pct(low),
        "avg_monthly_charges": round(results_df["Monthly Charges"].mean(), 2),
        "avg_tenure": round(results_df["Tenure"].mean(), 1),
        "at_risk_revenue": round(at_risk_rev, 2),
        "high_risk_revenue": round(high_rev, 2),
    }


def save_results_csv(results_df, filename, directory=None):
    directory = directory or REPORTS_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    results_df.to_csv(path, index=False)
    return path


def save_bundle_cache(bundle, cache_key, directory=None):
    directory = directory or REPORTS_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f".cache_{cache_key}.pkl")
    with open(path, "wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_bundle_cache(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        logger.warning("Could not load cache: %s", exc)
        return None
