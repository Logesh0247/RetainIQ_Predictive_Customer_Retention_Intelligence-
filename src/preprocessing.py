from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

feature_columns = joblib.load(FEATURE_PATH)


# -----------------------------------------
# Single Customer Prediction
# -----------------------------------------
def preprocess_input(form):

    row = {
        "Gender": form["gender"],
        "Senior Citizen": form["senior"],
        "Partner": form["partner"],
        "Dependents": form["dependents"],
        "Tenure Months": float(form["tenure"]),
        "Phone Service": form["phone"],
        "Multiple Lines": form["multiple_lines"],
        "Internet Service": form["internet"],
        "Online Security": form["online_security"],
        "Online Backup": form["online_backup"],
        "Device Protection": form["device_protection"],
        "Tech Support": form["tech_support"],
        "Streaming TV": form["streaming_tv"],
        "Streaming Movies": form["streaming_movies"],
        "Contract": form["contract"],
        "Paperless Billing": form["paperless"],
        "Payment Method": form["payment"],
        "Monthly Charges": float(form["monthly_charges"]),
        "Total Charges": float(form["total_charges"]),
    }

    df = pd.DataFrame([row])

    return preprocess_dataframe(df)


# -----------------------------------------
# Bulk Prediction
# -----------------------------------------
def preprocess_dataframe(df):

    data = pd.DataFrame(
        0,
        index=df.index,
        columns=feature_columns
    )

    # Numerical Columns
    
    data["Monthly Charges"] = df["Monthly Charges"].astype(float)
    
    data["Total Charges"] = (pd.to_numeric(df["Total Charges"],errors="coerce"))

    data["Total Charges"] = (data["Total Charges"].fillna(0))

    # Gender
    data.loc[df["Gender"] == "Male", "Gender_Male"] = 1

    # Senior Citizen
    data.loc[df["Senior Citizen"] == "Yes", "Senior Citizen_Yes"] = 1

    # Partner
    data.loc[df["Partner"] == "Yes", "Partner_Yes"] = 1

    # Dependents
    data.loc[df["Dependents"] == "Yes", "Dependents_Yes"] = 1

    # Phone Service
    data.loc[df["Phone Service"] == "Yes", "Phone Service_Yes"] = 1

    # Multiple Lines
    data.loc[
        df["Multiple Lines"] == "No phone service",
        "Multiple Lines_No phone service"
    ] = 1

    data.loc[
        df["Multiple Lines"] == "Yes",
        "Multiple Lines_Yes"
    ] = 1

    # Internet Service
    data.loc[
        df["Internet Service"] == "Fiber optic",
        "Internet Service_Fiber optic"
    ] = 1

    data.loc[
        df["Internet Service"] == "No",
        "Internet Service_No"
    ] = 1

    # Online Security
    data.loc[
        df["Online Security"] == "Yes",
        "Online Security_Yes"
    ] = 1

    data.loc[
        df["Online Security"] == "No internet service",
        "Online Security_No internet service"
    ] = 1

    # Online Backup
    data.loc[
        df["Online Backup"] == "Yes",
        "Online Backup_Yes"
    ] = 1

    data.loc[
        df["Online Backup"] == "No internet service",
        "Online Backup_No internet service"
    ] = 1

    # Device Protection
    data.loc[
        df["Device Protection"] == "Yes",
        "Device Protection_Yes"
    ] = 1

    data.loc[
        df["Device Protection"] == "No internet service",
        "Device Protection_No internet service"
    ] = 1

    # Tech Support
    data.loc[
        df["Tech Support"] == "Yes",
        "Tech Support_Yes"
    ] = 1

    data.loc[
        df["Tech Support"] == "No internet service",
        "Tech Support_No internet service"
    ] = 1

    # Streaming TV
    data.loc[
        df["Streaming TV"] == "Yes",
        "Streaming TV_Yes"
    ] = 1

    data.loc[
        df["Streaming TV"] == "No internet service",
        "Streaming TV_No internet service"
    ] = 1

    # Streaming Movies
    data.loc[
        df["Streaming Movies"] == "Yes",
        "Streaming Movies_Yes"
    ] = 1

    data.loc[
        df["Streaming Movies"] == "No internet service",
        "Streaming Movies_No internet service"
    ] = 1

    # Contract
    data.loc[
        df["Contract"] == "One year",
        "Contract_One year"
    ] = 1

    data.loc[
        df["Contract"] == "Two year",
        "Contract_Two year"
    ] = 1

    # Paperless Billing
    data.loc[
        df["Paperless Billing"] == "Yes",
        "Paperless Billing_Yes"
    ] = 1

    # Payment Method
    data.loc[
        df["Payment Method"] == "Credit card (automatic)",
        "Payment Method_Credit card (automatic)"
    ] = 1

    data.loc[
        df["Payment Method"] == "Electronic check",
        "Payment Method_Electronic check"
    ] = 1

    data.loc[
        df["Payment Method"] == "Mailed check",
        "Payment Method_Mailed check"
    ] = 1

    return data