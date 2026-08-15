"""
Bulk scoring adapter for the RetainIQ UI.
Uses the integrated project's Random Forest model and canonical preprocessing.
"""
import os, io, pickle, logging
import numpy as np, pandas as pd
from src.preprocessing import preprocess_bulk, PreprocessingError
from utils.prediction import load_model, calculate_risk_level, risk_css_class, signal_strength, get_top_drivers
from utils.recommendation import generate_recommendation, generate_recommendation_summary

logger=logging.getLogger("retainiq")
ALLOWED_EXTENSIONS={"csv"}; MAX_ROWS=20000; TOP_N_WITH_DRIVERS=10
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR=os.path.join(BASE_DIR,"reports"); UPLOADS_DIR=os.path.join(BASE_DIR,"uploads")
os.makedirs(REPORTS_DIR,exist_ok=True); os.makedirs(UPLOADS_DIR,exist_ok=True)

class BulkPredictionError(ValueError): pass
def allowed_file(filename): return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def _predict_batch(bundle,X): return bundle["model"].predict_proba(X)[:,1]

def run_bulk_prediction(file_storage):
    if file_storage is None or file_storage.filename=="": raise BulkPredictionError("No file was selected. Please choose a CSV file to upload.")
    if not allowed_file(file_storage.filename): raise BulkPredictionError("Invalid file type. Only .csv files are supported.")
    try:
        raw=file_storage.read()
        if not raw: raise BulkPredictionError("The uploaded file is empty.")
        df=pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError: raise BulkPredictionError("The uploaded CSV file has no data.")
    except pd.errors.ParserError: raise BulkPredictionError("The uploaded file could not be parsed as a valid CSV.")
    except BulkPredictionError: raise
    except Exception as exc: raise BulkPredictionError(f"Could not read the uploaded file: {exc}")
    if df.empty: raise BulkPredictionError("The uploaded CSV contains no customer rows.")
    if len(df)>MAX_ROWS: raise BulkPredictionError(f"The uploaded CSV has too many rows (max {MAX_ROWS:,}).")
    try: X,cleaned=preprocess_bulk(df)
    except PreprocessingError as exc: raise BulkPredictionError(str(exc))
    bundle=load_model()
    if bundle is None: raise BulkPredictionError("The churn prediction model is currently unavailable. Please make sure models/best_model.pkl exists.")
    try: probs=np.clip(_predict_batch(bundle,X),0,1)
    except Exception as exc:
        logger.exception("Bulk model prediction failed")
        raise BulkPredictionError(f"An error occurred while running predictions on this dataset: {exc}")
    results=pd.DataFrame({
        "Customer ID":cleaned["customerID"].values,
        "Churn Probability":np.round(probs*100,1),
        "Prediction":np.where(probs>=.5,"Likely to Churn","Likely to Stay"),
        "Risk Level":[calculate_risk_level(p) for p in probs],
        "Monthly Charges":cleaned["MonthlyCharges"].round(2).values,
        "Tenure":cleaned["tenure"].values,
        "Contract":cleaned["Contract"].values,
    })
    results["Recommendation"]=[generate_recommendation_summary(cleaned.iloc[i],probs[i],results["Risk Level"].iloc[i]) for i in range(len(cleaned))]
    return {"results_df":results,"cleaned_df":cleaned.reset_index(drop=True),"feature_frame":X.reset_index(drop=True),"probabilities":probs}

def build_display_records(bundle,sort_by="probability"):
    r,c,p=bundle["results_df"],bundle["cleaned_df"],bundle["probabilities"]; records=[]
    for i in range(len(r)):
        row=r.iloc[i]; cr=c.iloc[i]; prob=float(p[i]); risk=row["Risk Level"]
        records.append({"id":row["Customer ID"],"probability_pct":row["Churn Probability"],"prediction":row["Prediction"],
            "will_churn":row["Prediction"]=="Likely to Churn","risk_level":risk,"risk_css":risk_css_class(risk),
            "signal_bars":signal_strength(prob),"monthly_charges":row["Monthly Charges"],"tenure":row["Tenure"],
            "contract":row["Contract"],"recommendations":generate_recommendation(cr,prob,risk)})
    key={"probability":lambda x:-x["probability_pct"],"charges":lambda x:-x["monthly_charges"],"tenure":lambda x:x["tenure"]}.get(sort_by,lambda x:-x["probability_pct"])
    return sorted(records,key=key)

def build_top_risk_records(bundle,top_n=TOP_N_WITH_DRIVERS):
    r,c,X,p=bundle["results_df"],bundle["cleaned_df"],bundle["feature_frame"],bundle["probabilities"]
    records=[]
    for idx in np.argsort(-p)[:top_n]:
        idx=int(idx); row=r.iloc[idx]; cr=c.iloc[idx]; prob=float(p[idx]); risk=row["Risk Level"]
        records.append({"id":row["Customer ID"],"probability_pct":row["Churn Probability"],"risk_level":risk,
            "risk_css":risk_css_class(risk),"signal_bars":signal_strength(prob),"monthly_charges":row["Monthly Charges"],
            "tenure":row["Tenure"],"contract":row["Contract"],"internet_service":cr.get("InternetService"),
            "payment_method":cr.get("PaymentMethod"),"top_drivers":get_top_drivers(X.iloc[[idx]]),
            "recommendations":generate_recommendation(cr,prob,risk)})
    return records

def compute_dashboard_kpis(results_df):
    if results_df is None or results_df.empty:return None
    total=len(results_df); churners=int((results_df["Prediction"]=="Likely to Churn").sum())
    high=int((results_df["Risk Level"]=="High Risk").sum()); med=int((results_df["Risk Level"]=="Medium Risk").sum()); low=int((results_df["Risk Level"]=="Low Risk").sum())
    pct=lambda n:round(n/total*100,1) if total else 0.0
    return {"total_customers":total,"predicted_churners":churners,"churn_rate":round(churners/total*100,1),
        "avg_churn_probability":round(results_df["Churn Probability"].mean(),1),"high_risk_count":high,"medium_risk_count":med,"low_risk_count":low,
        "high_risk_pct":pct(high),"medium_risk_pct":pct(med),"low_risk_pct":pct(low),
        "avg_monthly_charges":round(results_df["Monthly Charges"].mean(),2),"avg_tenure":round(results_df["Tenure"].mean(),1),
        "at_risk_revenue":round(results_df.loc[results_df["Risk Level"].isin(["High Risk","Medium Risk"]),"Monthly Charges"].sum(),2)}

def save_results_csv(results_df,filename):
    path=os.path.join(REPORTS_DIR,filename); results_df.to_csv(path,index=False); return path
MAX_DISPLAY_ROWS=150
def save_bundle_cache(bundle,cache_key):
    path=os.path.join(REPORTS_DIR,f".cache_{cache_key}.pkl")
    with open(path,"wb") as f: pickle.dump(bundle,f)
    return path
def load_bundle_cache(path):
    if not path or not os.path.exists(path): return None
    try:
        with open(path,"rb") as f:return pickle.load(f)
    except Exception as exc: logger.warning("Could not load cache: %s",exc); return None
