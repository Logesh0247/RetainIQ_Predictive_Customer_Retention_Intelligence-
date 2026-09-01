"""
Bulk scoring adapter for the RetainIQ UI.
Uses the integrated project's Logistic Regression model and canonical preprocessing.
"""
import os
import io
import pickle
import logging
import tempfile
from collections import OrderedDict
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


def _writable_dir(preferred, fallback_name):
    """Return a directory we can actually write to.

    Managed hosts (Render, Fly, App Runner, containers with a read-only
    layer, ...) may not allow writes next to the source tree. Falling back to
    a temp directory keeps the app booting instead of crashing on import,
    which is the classic "works locally, 500s after deploy" failure.
    """
    candidates = [preferred]
    env_root = os.environ.get("RETAINIQ_DATA_DIR")
    if env_root:
        candidates.insert(0, os.path.join(env_root, fallback_name))
    candidates.append(os.path.join(tempfile.gettempdir(), "retainiq", fallback_name))
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write_test")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            return candidate
        except Exception as exc:  # pragma: no cover - platform dependent
            logging.getLogger("retainiq").warning(
                "Directory %s is not writable (%s); trying next location.", candidate, exc
            )
    return tempfile.mkdtemp(prefix=f"retainiq_{fallback_name}_")


REPORTS_DIR = _writable_dir(str(PATHS_REPORTS), "reports")
UPLOADS_DIR = _writable_dir(os.path.join(BASE_DIR, "uploads"), "uploads")

# Keeps the most recent scored runs in RAM as well as on disk. On hosts with an
# ephemeral or per-request filesystem the disk copy can vanish between
# requests; the memory copy keeps the dashboard/download links alive for as
# long as the worker is up.
_BUNDLE_MEMORY_CACHE = OrderedDict()
_BUNDLE_MEMORY_LIMIT = 3


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


def _read_csv_bytes(raw, filename="upload.csv"):
    """Parse an uploaded CSV with the same encoding fallbacks used for scoring."""
    if not raw:
        raise BulkPredictionError("The uploaded file is empty.")
    if filename and not allowed_file(filename):
        raise BulkPredictionError("Invalid file type. Please upload a CSV file.")
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
    if df is None:
        raise BulkPredictionError(
            "The file could not be read as a valid CSV. Check its formatting and try again."
        ) from last_err
    return df


def inspect_bulk_prediction_bytes(raw, filename="upload.csv"):
    """Return a safe, non-predictive dataset profile for the upload workspace."""
    from src.preprocessing import REQUIRED_COLUMNS, _normalize_columns

    df = _read_csv_bytes(raw, filename)
    if df.empty:
        raise BulkPredictionError("The uploaded CSV contains no customer rows.")
    if len(df) > MAX_ROWS:
        raise BulkPredictionError(f"The uploaded CSV has too many rows (max {MAX_ROWS:,}).")

    normalized = _normalize_columns(df)
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    required = [
        {"name": column, "present": column in normalized.columns}
        for column in REQUIRED_COLUMNS
    ]
    preview = df.head(5).copy()
    preview = preview.astype(object).where(pd.notna(preview), None)

    # Count missing values accurately: re-read with no aggressive NA detection,
    # then count only NaN cells. This avoids false positives from values like
    # "NA", "null", "None", or empty strings that are semantically valid.
    missing_count = 0
    try:
        df_profile = None
        for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252", "iso-8859-1"]:
            try:
                df_profile = pd.read_csv(io.BytesIO(raw), encoding=enc, keep_default_na=False, na_values=[])
                break
            except Exception:
                continue
        if df_profile is not None:
            missing_count = int(df_profile.isna().sum().sum())
    except Exception:
        missing_count = int(df.isna().sum().sum())

    return {
        "filename": os.path.basename(filename),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(column) for column in df.columns],
        "missing_values": missing_count,
        "duplicate_rows": int(df.duplicated().sum()),
        "required_columns": required,
        "missing_columns": missing,
        "valid": not missing,
        "preview_columns": [str(column) for column in preview.columns],
        "preview_rows": preview.to_dict("records"),
    }


def run_bulk_prediction_from_bytes(raw, filename="upload.csv"):
    return _score_dataframe(_read_csv_bytes(raw, filename))


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
    def _get(obj, key, default=""):
        if hasattr(obj, 'get'):
            return obj.get(key, default)
        try:
            return obj[key]
        except (KeyError, IndexError):
            return default

    risk = _get(row, "Risk Level", "Low Risk")
    m_charges = _get(row, "Monthly Charges", 0)
    c_id = _get(row, "Customer ID", "")
    pred = _get(row, "Prediction", "Likely to Stay")
    prob_pct = _get(row, "Churn Probability", 0)
    tenure_val = _get(row, "Tenure", 0)
    contract_val = _get(row, "Contract", "Month-to-month")

    if extra and "recommendations" in extra:
        recs = extra["recommendations"]
    else:
        try:
            recs = generate_recommendation(cleaned_row, prob, risk)
        except Exception:
            recs = None
    if not recs:
        recs = ["Monitor account."]

    record = {
        "id": str(c_id), "probability_pct": float(prob_pct), "prediction": str(pred),
        "will_churn": str(pred) == "Likely to Churn", "risk_level": str(risk),
        "risk_css": risk_css_class(str(risk)), "signal_bars": signal_strength(float(prob)),
        "monthly_charges": float(m_charges), "tenure": int(tenure_val),
        "contract": str(contract_val), "recommendations": recs, "primary_action": recs[0],
        "is_high": str(risk) == "High Risk", "is_mtm": str(contract_val) == "Month-to-month",
        "is_high_bill": float(m_charges) >= 80,
        # Data attributes for dynamic filtering
        "data_risk": str(risk).lower().replace(" ", "-"),
        "data_prediction": str(pred).lower().replace(" ", "-"),
        "data_contract": str(contract_val).lower().replace(" ", "-"),
        "data_charges": float(m_charges),
        "data_tenure": int(tenure_val),
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
    is_universal = bundle.get("is_universal", False)
    precomputed_recs = bundle.get("all_recommendations") if is_universal else None
    if sort_by == "charges" and "Monthly Charges" in results.columns:
        order = np.argsort(-results["Monthly Charges"].values)
    elif sort_by == "tenure" and "Tenure" in results.columns:
        order = np.argsort(results["Tenure"].values)
    else:
        order = np.argsort(-probs)
    top_indices = order[:max_rows]
    records = []
    for idx in top_indices:
        idx = int(idx)
        extra = None
        if precomputed_recs and idx < len(precomputed_recs):
            extra = {"recommendations": precomputed_recs[idx]}
        records.append(_record_from_row(results.iloc[idx], cleaned.iloc[idx], float(probs[idx]), extra=extra))
    return records


def _get_universal_drivers(bundle, idx):
    try:
        feature_names = bundle.get("feature_names", [])
        if not feature_names:
            return None
        # Get importances from model or from bundle (heuristic mode)
        model = bundle.get("model")
        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif bundle.get("feature_importances") is not None:
            importances = bundle["feature_importances"]
        else:
            return None
        feature_row = bundle["feature_frame"].iloc[[idx]]
        contributions = []
        for i, (name, importance) in enumerate(zip(feature_names, importances)):
            if i >= len(feature_row.columns):
                break
            value = feature_row.iloc[0, i]
            if importance > 0 and value != 0:
                contributions.append((name, float(importance * abs(value))))
        contributions.sort(key=lambda x: x[1], reverse=True)
        top_drivers = [name.replace("_", " ").title() for name, _ in contributions[:4]]
        return top_drivers if top_drivers else None
    except Exception as exc:
        logger.warning("Universal driver explanation failed: %s", exc)
        return None


def build_top_risk_records(bundle, top_n=TOP_N_WITH_DRIVERS):
    results, cleaned, features, probs = (
        bundle["results_df"], bundle["cleaned_df"], bundle["feature_frame"], bundle["probabilities"],
    )
    is_universal = bundle.get("is_universal", False)
    precomputed_recs = bundle.get("all_recommendations") if is_universal else None
    records = []
    top_indices = np.argsort(-probs)[:top_n]
    for idx in top_indices:
        idx = int(idx)
        risk = str(results.iloc[idx]["Risk Level"])
        if precomputed_recs and idx < len(precomputed_recs):
            recs = precomputed_recs[idx]
        elif is_universal:
            from utils.universal_churn import generate_universal_recommendations
            recs = generate_universal_recommendations(
                results.iloc[idx], float(probs[idx]), risk,
                feature_values=features.iloc[idx].values if hasattr(features, 'iloc') else None,
                feature_names=bundle.get("feature_names"),
                feature_importances=bundle.get("feature_importances"),
                feature_stats=bundle.get("feature_stats"),
            )
        else:
            recs = generate_recommendation(cleaned.iloc[idx], float(probs[idx]), risk)
        if not recs:
            recs = ["Monitor account."]
        if is_universal:
            top_drivers = _get_universal_drivers(bundle, idx)
        else:
            top_drivers = get_top_drivers(features.iloc[[idx]])
        records.append(_record_from_row(
            results.iloc[idx], cleaned.iloc[idx], float(probs[idx]),
            extra={
                "internet_service": cleaned.iloc[idx].get("InternetService", "N/A") if "InternetService" in cleaned.columns else "N/A",
                "payment_method": cleaned.iloc[idx].get("PaymentMethod", "N/A") if "PaymentMethod" in cleaned.columns else "N/A",
                "top_drivers": top_drivers, "recommendations": recs, "primary_action": recs[0],
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
    charges = pd.to_numeric(results_df["Monthly Charges"], errors='coerce').fillna(0)
    tenure = pd.to_numeric(results_df["Tenure"], errors='coerce').fillna(0)
    high_rev = float(charges[results_df["Risk Level"] == "High Risk"].sum())
    at_risk_mask = results_df["Risk Level"].isin(["High Risk", "Medium Risk"])
    at_risk_rev = float(charges[at_risk_mask].sum())
    return {
        "total_customers": total,
        "predicted_churners": churners,
        "churn_rate": round(churners / total * 100, 1),
        "avg_churn_probability": round(pd.to_numeric(results_df["Churn Probability"], errors='coerce').mean(), 1),
        "high_risk_count": high,
        "medium_risk_count": med,
        "low_risk_count": low,
        "high_risk_pct": pct(high),
        "medium_risk_pct": pct(med),
        "low_risk_pct": pct(low),
        "avg_monthly_charges": round(charges.mean(), 2),
        "avg_tenure": round(tenure.mean(), 1),
        "at_risk_revenue": round(at_risk_rev, 2),
        "high_risk_revenue": round(high_rev, 2),
    }


def _series_analysis(df, labels, groups):
    """Return actual customer counts and predicted churn rates per business segment."""
    rows = []
    for label, mask in zip(labels, groups):
        segment = df.loc[mask]
        rows.append({
            "label": label,
            "count": int(len(segment)),
            "rate": round(float((segment["Prediction"] == "Likely to Churn").mean() * 100), 1)
                    if len(segment) else 0.0,
        })
    return rows


def build_dashboard_analytics(results_df):
    """Build chart-ready summaries exclusively from scored result rows."""
    if results_df is None or results_df.empty:
        return {"probability": [], "contract": [], "tenure": [], "charges": []}
    probability = results_df["Churn Probability"].astype(float)
    if "Contract" in results_df.columns:
        contract_labels = results_df["Contract"].value_counts().head(5).index.tolist()
        contract_masks = [results_df["Contract"] == v for v in contract_labels]
        contract = _series_analysis(results_df, contract_labels, contract_masks)
    else:
        contract = []
    if "Tenure" in results_df.columns:
        tenure = results_df["Tenure"].astype(float)
        tenure_analysis = _series_analysis(results_df,
            ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo", "73+ mo"],
            [tenure <= 12, (tenure > 12) & (tenure <= 24), (tenure > 24) & (tenure <= 48), (tenure > 48) & (tenure <= 72), tenure > 72])
    else:
        tenure_analysis = []
    if "Monthly Charges" in results_df.columns:
        charges = results_df["Monthly Charges"].astype(float)
        charges_analysis = _series_analysis(results_df,
            ["<$40", "$40–69", "$70–99", "$100+"],
            [charges < 40, (charges >= 40) & (charges < 70), (charges >= 70) & (charges < 100), charges >= 100])
    else:
        charges_analysis = []
    return {
        "probability": [
            {"label": label, "count": int(mask.sum()), "pct": round(float(mask.sum()) / len(results_df) * 100, 1)}
            for label, mask in zip(["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
                [probability < 20, (probability >= 20) & (probability < 40), (probability >= 40) & (probability < 60), (probability >= 60) & (probability < 80), probability >= 80])
        ],
        "contract": contract, "tenure": tenure_analysis, "charges": charges_analysis,
    }


def build_dashboard_page(bundle, risk="All", prediction="All", contract="All", search="",
                         page=1, page_size=10):
    """Filter and paginate scored customers without sending large datasets to the browser."""
    results = bundle["results_df"]
    mask = pd.Series(True, index=results.index)
    if risk in ("High Risk", "Medium Risk", "Low Risk"):
        mask &= results["Risk Level"] == risk
    if prediction in ("Likely to Churn", "Likely to Stay"):
        mask &= results["Prediction"] == prediction
    if contract in ("Month-to-month", "One year", "Two year") and "Contract" in results.columns:
        mask &= results["Contract"] == contract
    if search:
        mask &= results["Customer ID"].astype(str).str.contains(str(search), case=False, regex=False)

    filtered = results.loc[mask].sort_values("Churn Probability", ascending=False)
    total = len(filtered)
    pages = max(1, int(np.ceil(total / page_size)))
    page = max(1, min(int(page), pages))
    indices = filtered.index[(page - 1) * page_size:page * page_size]
    is_universal = bundle.get("is_universal", False)
    precomputed_recs = bundle.get("all_recommendations") if is_universal else None
    rows = []
    for idx in indices:
        row = results.loc[idx]
        prob = float(bundle["probabilities"][idx])
        extra = None
        if precomputed_recs and idx < len(precomputed_recs):
            extra = {"recommendations": precomputed_recs[idx]}
        record = _record_from_row(row, bundle["cleaned_df"].loc[idx], prob, extra=extra)
        rows.append(record)
    return {
        "rows": rows,
        "total": int(total),
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "kpis": compute_dashboard_kpis(filtered),
        "analytics": build_dashboard_analytics(filtered),
    }


def build_customer_detail(bundle, customer_id):
    """Return one actual scored customer with model drivers and profile attributes."""
    ids = bundle["results_df"]["Customer ID"].astype(str)
    matches = bundle["results_df"].index[ids.str.casefold() == str(customer_id).strip().casefold()]
    if len(matches) == 0:
        return None
    idx = int(matches[0])
    result = bundle["results_df"].loc[idx]
    cleaned = bundle["cleaned_df"].loc[idx]
    prob = float(bundle["probabilities"][idx])
    is_universal = bundle.get("is_universal", False)
    extra = {}
    if is_universal:
        # For universal bundles, use precomputed recommendations and drivers
        precomputed_recs = bundle.get("all_recommendations")
        if precomputed_recs and idx < len(precomputed_recs):
            extra["recommendations"] = precomputed_recs[idx]
        # Get universal drivers
        extra["top_drivers"] = _get_universal_drivers(bundle, idx)
        # Add generic profile attributes from cleaned data
        for col in cleaned.index[:8]:
            extra[str(col)] = str(cleaned[col])
    else:
        # Telco-specific attributes
        extra.update({
            "internet_service": str(cleaned.get("InternetService", "Unavailable")),
            "payment_method": str(cleaned.get("PaymentMethod", "Unavailable")),
            "gender": str(cleaned.get("gender", "Unavailable")),
            "senior_citizen": "Yes" if int(cleaned.get("SeniorCitizen", 0)) == 1 else "No",
            "top_drivers": get_top_drivers(bundle["feature_frame"].iloc[[idx]]),
        })
    return _record_from_row(result, cleaned, prob, extra=extra)


def save_results_csv(results_df, filename, directory=None):
    directory = directory or REPORTS_DIR
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        results_df.to_csv(path, index=False)
        return path
    except OSError as exc:
        # Read-only or full filesystem: keep serving the results from memory
        # instead of failing the whole prediction.
        logger.warning("Could not write results CSV to %s: %s", directory, exc)
        return os.path.join(directory, filename)


def _remember_bundle(cache_key, bundle):
    _BUNDLE_MEMORY_CACHE[cache_key] = bundle
    _BUNDLE_MEMORY_CACHE.move_to_end(cache_key)
    while len(_BUNDLE_MEMORY_CACHE) > _BUNDLE_MEMORY_LIMIT:
        _BUNDLE_MEMORY_CACHE.popitem(last=False)


def save_bundle_cache(bundle, cache_key, directory=None):
    directory = directory or REPORTS_DIR
    path = os.path.join(directory, f".cache_{cache_key}.pkl")
    _remember_bundle(cache_key, bundle)
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("Could not persist bundle cache to %s: %s", path, exc)
    return path


def _cache_key_from_path(path):
    name = os.path.basename(path or "")
    if name.startswith(".cache_") and name.endswith(".pkl"):
        return name[len(".cache_"):-len(".pkl")]
    return name


def load_bundle_cache(path):
    """Load a scored run: disk first, then the in-process memory cache."""
    cache_key = _cache_key_from_path(path)
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                bundle = pickle.load(handle)
            if cache_key:
                _remember_bundle(cache_key, bundle)
            return bundle
        except Exception as exc:
            logger.warning("Could not load cache: %s", exc)
    if cache_key and cache_key in _BUNDLE_MEMORY_CACHE:
        logger.info("Serving run %s from the in-memory cache.", cache_key)
        _BUNDLE_MEMORY_CACHE.move_to_end(cache_key)
        return _BUNDLE_MEMORY_CACHE[cache_key]
    return None

