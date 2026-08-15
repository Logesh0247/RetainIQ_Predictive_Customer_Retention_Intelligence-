"""
RetainIQ preprocessing adapter.

The UI uses the canonical Telco dataset field names (camelCase/raw Kaggle names),
while the deployed Random Forest model in the integrated project was trained on
the legacy display-name feature schema. This module converts either schema into
the exact 30 features expected by models/feature_columns.pkl.
"""
import pandas as pd
import numpy as np
import joblib
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

# Legacy schema -> canonical raw schema
LEGACY_TO_RAW = {
    "CustomerID":"customerID", "Customer ID":"customerID",
    "Gender":"gender", "Senior Citizen":"SeniorCitizen",
    "Tenure Months":"tenure", "Phone Service":"PhoneService",
    "Multiple Lines":"MultipleLines", "Internet Service":"InternetService",
    "Online Security":"OnlineSecurity", "Online Backup":"OnlineBackup",
    "Device Protection":"DeviceProtection", "Tech Support":"TechSupport",
    "Streaming TV":"StreamingTV", "Streaming Movies":"StreamingMovies",
    "Paperless Billing":"PaperlessBilling", "Payment Method":"PaymentMethod",
    "Monthly Charges":"MonthlyCharges", "Total Charges":"TotalCharges",
}

class PreprocessingError(ValueError):
    pass

def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {c: LEGACY_TO_RAW[c] for c in df.columns if c in LEGACY_TO_RAW}
    return df.rename(columns=rename)

def _clean_yes_no(v):
    if pd.isna(v): return "No"
    s = str(v).strip()
    return s if s in ("Yes","No") else "No"

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
    df.loc[blank, "customerID"] = [f"ROW-{i+1}" for i in range(int(blank.sum()))]

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).clip(0,100).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if df["MonthlyCharges"].notna().any():
        df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
    else:
        df["MonthlyCharges"] = 65.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int).clip(0,1)

    for col in ["Partner","Dependents","PhoneService","PaperlessBilling"]:
        df[col] = df[col].apply(_clean_yes_no)

    defaults = {
        "gender":"Female","MultipleLines":"No","InternetService":"No",
        "OnlineSecurity":"No internet service","OnlineBackup":"No internet service",
        "DeviceProtection":"No internet service","TechSupport":"No internet service",
        "StreamingTV":"No internet service","StreamingMovies":"No internet service",
        "Contract":"Month-to-month","PaymentMethod":"Mailed check",
    }
    for col, default in defaults.items():
        df[col] = df[col].fillna(default).astype(str).str.strip()
        df.loc[df[col].eq(""), col] = default
    return df

def build_feature_frame(df):
    df = clean_raw_dataframe(df) if not set(REQUIRED_COLUMNS).issubset(df.columns) else df.copy()
    # If passed a canonical cleaned dataframe, preserve it.
    if "customerID" not in df.columns:
        df["customerID"] = [f"ROW-{i+1}" for i in range(len(df))]
    data = pd.DataFrame(0, index=df.index, columns=FEATURE_COLUMNS)
    data["Tenure Months"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
    data["Monthly Charges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0).astype(float)
    data["Total Charges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype(float)

    def yes(col, feature):
        if feature in data.columns:
            data.loc[df[col].eq("Yes"), feature] = 1
    yes("gender", "Gender_Male")
    yes("SeniorCitizen", "Senior Citizen_Yes")
    for c in ["Partner","Dependents","PhoneService"]:
        yes(c, f"{c}_Yes")

    mappings = [
        ("MultipleLines","No phone service","Multiple Lines_No phone service"),
        ("MultipleLines","Yes","Multiple Lines_Yes"),
        ("InternetService","Fiber optic","Internet Service_Fiber optic"),
        ("InternetService","No","Internet Service_No"),
        ("OnlineSecurity","No internet service","Online Security_No internet service"),
        ("OnlineSecurity","Yes","Online Security_Yes"),
        ("OnlineBackup","No internet service","Online Backup_No internet service"),
        ("OnlineBackup","Yes","Online Backup_Yes"),
        ("DeviceProtection","No internet service","Device Protection_No internet service"),
        ("DeviceProtection","Yes","Device Protection_Yes"),
        ("TechSupport","No internet service","Tech Support_No internet service"),
        ("TechSupport","Yes","Tech Support_Yes"),
        ("StreamingTV","No internet service","Streaming TV_No internet service"),
        ("StreamingTV","Yes","Streaming TV_Yes"),
        ("StreamingMovies","No internet service","Streaming Movies_No internet service"),
        ("StreamingMovies","Yes","Streaming Movies_Yes"),
        ("Contract","One year","Contract_One year"),
        ("Contract","Two year","Contract_Two year"),
        ("PaperlessBilling","Yes","Paperless Billing_Yes"),
        ("PaymentMethod","Credit card (automatic)","Payment Method_Credit card (automatic)"),
        ("PaymentMethod","Electronic check","Payment Method_Electronic check"),
        ("PaymentMethod","Mailed check","Payment Method_Mailed check"),
    ]
    for raw, value, feature in mappings:
        if feature in data.columns:
            data.loc[df[raw].eq(value), feature] = 1
    return data[FEATURE_COLUMNS]

def validate_raw_record(record):
    record = {str(k): v for k,v in record.items()}
    missing = [c for c in REQUIRED_COLUMNS if c not in record or record[c] in (None,"")]
    missing = [m for m in missing if m != "TotalCharges"]
    if missing:
        raise PreprocessingError("Missing required field(s): " + ", ".join(missing))
    try:
        tenure = int(float(record["tenure"]))
    except Exception:
        raise PreprocessingError("'tenure' must be a whole number of months.")
    if tenure < 0 or tenure > 100:
        raise PreprocessingError("'tenure' must be between 0 and 100 months.")
    try: float(record["MonthlyCharges"])
    except Exception: raise PreprocessingError("'MonthlyCharges' must be numeric.")

def preprocess_single(record):
    validate_raw_record(record)
    raw = pd.DataFrame([record])
    cleaned = clean_raw_dataframe(raw)
    return build_feature_frame(cleaned), cleaned.iloc[0]

def preprocess_bulk(df):
    cleaned = clean_raw_dataframe(df)
    return build_feature_frame(cleaned), cleaned
