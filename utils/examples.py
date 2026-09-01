"""
Example customers for the Single Prediction page.

A 19-field blank form is a bad first impression: the visitor has to invent a
plausible telecom customer before the product does anything. These helpers pull
real rows out of the bundled sample portfolio, score them once, and expose a
high-risk / low-risk / typical / random example that pre-fills the form.
"""
import logging
import random

import pandas as pd

from src.paths import SAMPLE_CUSTOMERS
from src.preprocessing import RAW_COLUMNS, clean_raw_dataframe
from utils.prediction import load_model, _predict_proba
from src.preprocessing import build_feature_frame

logger = logging.getLogger("retainiq")

EXAMPLE_KINDS = ("high", "low", "typical", "random")

# Used when the bundled sample file is unavailable (trimmed deployment, etc.).
FALLBACK_EXAMPLE = {
    "customerID": "DEMO-0001",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": "3",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": "94.40",
    "TotalCharges": "283.20",
}

_CACHE = {}


def _row_to_form(row):
    form = {}
    for column in RAW_COLUMNS:
        value = row.get(column, "")
        if pd.isna(value):
            value = ""
        if column in ("MonthlyCharges", "TotalCharges"):
            form[column] = f"{float(value):.2f}" if value != "" else ""
        elif column in ("tenure", "SeniorCitizen"):
            form[column] = str(int(float(value))) if value != "" else "0"
        else:
            form[column] = str(value)
    return form


def _scored_examples():
    """Score the sample portfolio once and cache the interesting rows."""
    if "examples" in _CACHE:
        return _CACHE["examples"]

    examples = {}
    try:
        bundle = load_model()
        df = pd.read_csv(SAMPLE_CUSTOMERS)
        cleaned = clean_raw_dataframe(df)
        if bundle is not None and not cleaned.empty:
            frame = build_feature_frame(cleaned)
            probabilities = bundle["model"].predict_proba(frame)[:, 1]
            cleaned = cleaned.assign(_probability=probabilities)
            ordered = cleaned.sort_values("_probability")
            examples = {
                "low": _row_to_form(ordered.iloc[0]),
                "high": _row_to_form(ordered.iloc[-1]),
                "typical": _row_to_form(ordered.iloc[len(ordered) // 2]),
                "_pool": cleaned,
            }
    except Exception as exc:  # pragma: no cover - depends on bundled data file
        logger.warning("Could not build example customers: %s", exc)
        examples = {}

    _CACHE["examples"] = examples
    return examples


def get_example(kind="high"):
    """Return form-ready field values for the requested example customer."""
    kind = (kind or "high").strip().lower()
    if kind not in EXAMPLE_KINDS:
        kind = "high"

    examples = _scored_examples()
    if not examples:
        return dict(FALLBACK_EXAMPLE)

    if kind == "random":
        pool = examples.get("_pool")
        if pool is None or pool.empty:
            return dict(FALLBACK_EXAMPLE)
        return _row_to_form(pool.iloc[random.randrange(len(pool))])

    return dict(examples.get(kind) or FALLBACK_EXAMPLE)
