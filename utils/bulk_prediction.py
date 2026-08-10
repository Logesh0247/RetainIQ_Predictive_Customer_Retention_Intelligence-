import joblib
import pandas as pd

from pathlib import Path
from src.preprocessing import preprocess_dataframe
from src.recommendation import get_risk, get_recommendation

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_bulk_customers(csv_path, output_folder):
    """
    Predict churn for all customers in an uploaded CSV.
    """

    # -------------------------
    # Read CSV
    # -------------------------
    df = pd.read_csv(csv_path)

    # -------------------------
    # Store Customer IDs
    # -------------------------
    if "CustomerID" in df.columns:
        customer_ids = df["CustomerID"]
    elif "customerID" in df.columns:
        customer_ids = df["customerID"]
    else:
        customer_ids = pd.Series(range(1, len(df) + 1),name="CustomerID")

    # -------------------------
    # Required columns
    # -------------------------
    required_columns = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns]

    if missing_columns:
        raise Exception("Missing columns:\n\n"+ "\n".join(missing_columns))

    # -------------------------
    # Preprocess
    # -------------------------
    processed_df = preprocess_dataframe(df[required_columns])

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(processed_df)

    probability = model.predict_proba(processed_df)[:, 1]

    # -------------------------
    # Result DataFrame
    # -------------------------
    result = pd.DataFrame()

    result["CustomerID"] = customer_ids

    result["Prediction"] = [
        "Customer Will Churn"
        if p == 1
        else "Customer Will Stay"
        for p in prediction]

    result["Probability"] = (probability * 100).round(2)

    result["Risk"] = [
        get_risk(p)
        for p in probability]

    recommendations = []

    for index, row in df.iterrows():

        risk = result.loc[index, "Risk"]

        recommendations.append(

            get_recommendation(

                row,

                risk

            )

        )

    result["Recommendation"] = recommendations

    # -------------------------
    # KPIs
    # -------------------------
    
    total = len(result)

    churn = len(result[result["Prediction"] == "Customer Will Churn"])

    stay = len(result[result["Prediction"] == "Customer Will Stay"])
    
    churn_rate = round((churn / total) * 100, 2)

    high = len(result[result["Risk"] == "High"])

    medium = len(result[result["Risk"] == "Medium"])

    low = len(result[result["Risk"] == "Low"])

    avg_probability = round(result["Probability"].mean(), 2)

    avg_monthly_charges = round(df["Monthly Charges"].mean(),2)

    revenue_at_risk = round(df.loc[result["Prediction"] == "Customer Will Churn","Monthly Charges"].astype(float).sum(),2)

    critical_customers = len(result[result["Probability"] >= 90])

    avg_tenure = round(df["Tenure Months"].mean(),2)

    top10 = result.sort_values(by="Probability",ascending=False).head(10)
    
    retention_rate = round((stay / total) * 100,2)

    # -------------------------
    # Save CSV
    # -------------------------
    output_file = (Path(output_folder)/ "bulk_prediction_results.csv")

    result.to_csv(output_file,index=False,)

    # -------------------------
    # Return
    # -------------------------
    
    return {

    "table": result.to_dict(orient="records"),

    "top10": top10.to_dict(orient="records"),

    "total": total,

    "churn": churn,

    "churn_rate": churn_rate,

    "retention_rate": retention_rate,

    "stay": stay,

    "high": high,

    "medium": medium,

    "low": low,

    "avg_probability": avg_probability,
    
    "avg_monthly_charges": round(df["Monthly Charges"].astype(float).mean(),2),
    
    "revenue_at_risk": revenue_at_risk,

    "critical_customers": critical_customers,

    "avg_tenure": round(df["Tenure Months"].astype(float).mean(),2),
    
    "download_file": output_file.name,
}