"""
RetainIQ preprocessing adapter.

The UI uses the canonical Telco dataset field names (camelCase/raw Kaggle names),
while the deployed Logistic Regression model in the integrated project was trained on
the legacy display-name feature schema. This module converts either schema into
the exact 30 features expected by models/feature_columns.pkl.
"""
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"
FEATURE_COLUMNS = joblib.load(FEATURE_PATH)

RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
REQUIRED_COLUMNS = [c for c in RAW_COLUMNS if c != "customerID"]

# Legacy / alternative schema -> canonical raw schema
LEGACY_TO_RAW = {
    "CustomerID": "customerID", "Customer ID": "customerID", "customer_id": "customerID", "id": "customerID", "ID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen", "senior_citizen": "SeniorCitizen", "SeniorCitizen": "SeniorCitizen",
    "Tenure Months": "tenure", "Tenure": "tenure", "tenure_months": "tenure",
    "Phone Service": "PhoneService", "phone_service": "PhoneService",
    "Multiple Lines": "MultipleLines", "multiple_lines": "MultipleLines",
    "Internet Service": "InternetService", "internet_service": "InternetService",
    "Online Security": "OnlineSecurity", "online_security": "OnlineSecurity",
    "Online Backup": "OnlineBackup", "online_backup": "OnlineBackup",
    "Device Protection": "DeviceProtection", "device_protection": "DeviceProtection",
    "Tech Support": "TechSupport", "tech_support": "TechSupport",
    "Streaming TV": "StreamingTV", "streaming_tv": "StreamingTV",
    "Streaming Movies": "StreamingMovies", "streaming_movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling", "paperless_billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod", "payment_method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges", "monthly_charges": "MonthlyCharges",
    "Total Charges": "TotalCharges", "total_charges": "TotalCharges",
}

CANONICAL_LOOKUP = {
    "customerid": "customerID",
    "customer_id": "customerID",
    "id": "customerID",
    "gender": "gender",
    "seniorcitizen": "SeniorCitizen",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "tenuremonths": "tenure",
    "phoneservice": "PhoneService",
    "multiplelines": "MultipleLines",
    "internetservice": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "techsupport": "TechSupport",
    "streamingtv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "contract": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "monthlycharges": "MonthlyCharges",
    "totalcharges": "TotalCharges",
}

class PreprocessingError(ValueError):
    pass


def _normalize_columns(df):
    df = df.copy()
    rename = {}
    for c in df.columns:
        c_str = str(c).strip().strip("\ufeff")
        if c_str in LEGACY_TO_RAW:
            rename[c] = LEGACY_TO_RAW[c_str]
        else:
            c_clean = re.sub(r"[_\s\-]+", "", c_str).lower()
            if c_clean in CANONICAL_LOOKUP:
                rename[c] = CANONICAL_LOOKUP[c_clean]
    return df.rename(columns=rename)


def _clean_numeric_series(s, default_val=0.0):
    if s is None:
        return pd.Series(default_val)
    if isinstance(s, (int, float)):
        return pd.Series(float(s))
    if hasattr(s, "astype"):
        s_str = s.astype(str).str.replace(r"[\$\,\€\£\s]", "", regex=True)
        return pd.to_numeric(s_str, errors="coerce")
    try:
        cleaned = re.sub(r"[\$\,\€\£\s]", "", str(s))
        return pd.Series(float(cleaned))
    except Exception:
        return pd.Series(default_val)


def _clean_yes_no(v):
    if pd.isna(v):
        return "No"
    s = str(v).strip().lower()
    if s in ("yes", "y", "1", "true", "t"):
        return "Yes"
    if s in ("no", "n", "0", "false", "f"):
        return "No"
    return "No"


def _clean_contract(v):
    if pd.isna(v):
        return "Month-to-month"
    s = str(v).strip().lower()
    if any(k in s for k in ("two", "2")):
        return "Two year"
    if any(k in s for k in ("one", "1")):
        return "One year"
    return "Month-to-month"


def _clean_payment_method(v):
    if pd.isna(v):
        return "Mailed check"
    s = str(v).strip().lower()
    if "electronic" in s or "e-check" in s or "echeck" in s:
        return "Electronic check"
    if "credit" in s:
        return "Credit card (automatic)"
    if "bank" in s or "transfer" in s:
        return "Bank transfer (automatic)"
    if "mail" in s or "check" in s:
        return "Mailed check"
    return "Mailed check"


def _clean_internet_service(v):
    if pd.isna(v):
        return "No"
    s = str(v).strip().lower()
    if "fiber" in s:
        return "Fiber optic"
    if "dsl" in s:
        return "DSL"
    if "no" in s or "none" in s:
        return "No"
    return "DSL"


def clean_raw_dataframe(df):
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PreprocessingError(
            "The uploaded CSV is missing required column(s): " + ", ".join(missing)
        )
    if df.empty:
        raise PreprocessingError("The uploaded CSV contains no customer rows.")

    if "customerID" not in df.columns:
        df["customerID"] = [f"ROW-{i+1}" for i in range(len(df))]
    df["customerID"] = df["customerID"].fillna("").astype(str)
    blank = df["customerID"].str.strip().eq("")
    if blank.any():
        df.loc[blank, "customerID"] = [f"ROW-{i+1}" for i in range(int(blank.sum()))]

    df["tenure"] = _clean_numeric_series(df["tenure"]).fillna(0).clip(0, 100).astype(int)
    
    monthly_num = _clean_numeric_series(df["MonthlyCharges"])
    if monthly_num.notna().any():
        df["MonthlyCharges"] = monthly_num.fillna(monthly_num.median())
    else:
        df["MonthlyCharges"] = 65.0

    total_num = _clean_numeric_series(df["TotalCharges"])
    df["TotalCharges"] = total_num.fillna(df["tenure"] * df["MonthlyCharges"])

    senior_num = _clean_numeric_series(df["SeniorCitizen"])
    df["SeniorCitizen"] = senior_num.fillna(0).astype(int).clip(0, 1)

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].apply(_clean_yes_no)

    df["gender"] = df["gender"].fillna("Female").astype(str).str.strip().str.capitalize()
    df.loc[~df["gender"].isin(["Male", "Female"]), "gender"] = "Female"

    df["Contract"] = df["Contract"].apply(_clean_contract)
    df["PaymentMethod"] = df["PaymentMethod"].apply(_clean_payment_method)
    df["InternetService"] = df["InternetService"].apply(_clean_internet_service)

    defaults = {
        "MultipleLines": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
    }
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str).str.strip()
            df.loc[df[col].eq(""), col] = default
        else:
            df[col] = default

    return df


def build_feature_frame(df):
    df = clean_raw_dataframe(df) if not set(REQUIRED_COLUMNS).issubset(df.columns) else df.copy()
    if "customerID" not in df.columns:
        df["customerID"] = [f"ROW-{i+1}" for i in range(len(df))]

    data = pd.DataFrame(0, index=df.index, columns=FEATURE_COLUMNS)
    data["Tenure Months"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
    data["Monthly Charges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0).astype(float)
    data["Total Charges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype(float)

    def flag(mask, feature):
        if feature in data.columns:
            data.loc[mask, feature] = 1

    gender = df["gender"].astype(str).str.strip()
    flag(gender.str.lower().eq("male"), "Gender_Male")

    senior = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0)
    flag(senior.eq(1) | df["SeniorCitizen"].astype(str).str.strip().isin(("1", "Yes", "yes")), "Senior Citizen_Yes")

    flag(df["Partner"].eq("Yes"), "Partner_Yes")
    flag(df["Dependents"].eq("Yes"), "Dependents_Yes")
    flag(df["PhoneService"].eq("Yes"), "Phone Service_Yes")

    mappings = [
        ("MultipleLines", "No phone service", "Multiple Lines_No phone service"),
        ("MultipleLines", "Yes", "Multiple Lines_Yes"),
        ("InternetService", "Fiber optic", "Internet Service_Fiber optic"),
        ("InternetService", "No", "Internet Service_No"),
        ("OnlineSecurity", "No internet service", "Online Security_No internet service"),
        ("OnlineSecurity", "Yes", "Online Security_Yes"),
        ("OnlineBackup", "No internet service", "Online Backup_No internet service"),
        ("OnlineBackup", "Yes", "Online Backup_Yes"),
        ("DeviceProtection", "No internet service", "Device Protection_No internet service"),
        ("DeviceProtection", "Yes", "Device Protection_Yes"),
        ("TechSupport", "No internet service", "Tech Support_No internet service"),
        ("TechSupport", "Yes", "Tech Support_Yes"),
        ("StreamingTV", "No internet service", "Streaming TV_No internet service"),
        ("StreamingTV", "Yes", "Streaming TV_Yes"),
        ("StreamingMovies", "No internet service", "Streaming Movies_No internet service"),
        ("StreamingMovies", "Yes", "Streaming Movies_Yes"),
        ("Contract", "One year", "Contract_One year"),
        ("Contract", "Two year", "Contract_Two year"),
        ("PaperlessBilling", "Yes", "Paperless Billing_Yes"),
        ("PaymentMethod", "Credit card (automatic)", "Payment Method_Credit card (automatic)"),
        ("PaymentMethod", "Electronic check", "Payment Method_Electronic check"),
        ("PaymentMethod", "Mailed check", "Payment Method_Mailed check"),
    ]
    for raw, value, feature in mappings:
        if feature in data.columns:
            data.loc[df[raw].eq(value), feature] = 1
    return data[FEATURE_COLUMNS]


def validate_raw_record(record):
    record = {str(k): v for k, v in record.items()}
    missing = [c for c in REQUIRED_COLUMNS if c not in record or record[c] in (None, "")]
    missing = [m for m in missing if m != "TotalCharges"]
    if missing:
        raise PreprocessingError("Missing required field(s): " + ", ".join(missing))
    try:
        tenure = int(float(record["tenure"]))
    except Exception:
        raise PreprocessingError("'tenure' must be a whole number of months.")
    if tenure < 0 or tenure > 100:
        raise PreprocessingError("'tenure' must be between 0 and 100 months.")
    try:
        float(record["MonthlyCharges"])
    except Exception:
        raise PreprocessingError("'MonthlyCharges' must be numeric.")


def preprocess_single(record):
    validate_raw_record(record)
    raw = pd.DataFrame([record])
    cleaned = clean_raw_dataframe(raw)
    return build_feature_frame(cleaned), cleaned.iloc[0]


def preprocess_bulk(df):
    cleaned = clean_raw_dataframe(df)
    return build_feature_frame(cleaned), cleaned
