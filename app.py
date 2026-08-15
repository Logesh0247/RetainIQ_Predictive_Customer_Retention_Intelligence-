"""
app.py
------
RetainIQ -- Customer Churn Prediction & Retention Intelligence Platform.

Flask entrypoint. Routes stay thin: validation happens in src/preprocessing,
ML logic happens in utils/prediction & utils/bulk_prediction, recommendation
logic happens in utils/recommendation. This keeps the app modular and easy
to integrate into a larger existing RetainIQ system later.
"""

import os
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, abort, session
)

from src.preprocessing import PreprocessingError
from utils.prediction import (
    predict_customer, is_model_available, ModelLoadError, load_model,
)
from utils.bulk_prediction import (
    run_bulk_prediction, compute_dashboard_kpis, save_results_csv,
    build_display_records, build_top_risk_records, save_bundle_cache,
    load_bundle_cache, BulkPredictionError, MAX_DISPLAY_ROWS, REPORTS_DIR,
)
from utils.recommendation import generate_recommendation

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("RETAINIQ_SECRET_KEY", "retainiq-dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["REPORTS_FOLDER"] = REPORTS_DIR

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORTS_FOLDER"], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retainiq")

# Power BI embed configuration -- left blank until a real workspace URL
# is supplied. The app must not invent one.
POWERBI_URL = ""

FORM_FIELDS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}


# ---------------------------------------------------------------------------
# Error handlers -- never expose raw tracebacks
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="That signal doesn't exist. The page you're looking for isn't here."), 404


@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", code=413, message="The uploaded file is too large (max 10 MB)."), 413


@app.errorhandler(500)
def server_error(e):
    logger.exception("Unhandled server error")
    return render_template("error.html", code=500, message="Something went wrong on our end. Please try again."), 500


# ---------------------------------------------------------------------------
# Helper: retrieve the most recent bulk-prediction bundle for this session
# ---------------------------------------------------------------------------
def _get_latest_bundle():
    path = session.get("latest_bundle_path")
    return load_bundle_cache(path) if path else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", model_ready=is_model_available())


@app.route("/single-prediction", methods=["GET", "POST"])
def single_prediction():
    if request.method == "GET":
        return render_template("single_prediction.html", fields=FORM_FIELDS, form_data={}, errors=None)

    form_data = request.form.to_dict()
    try:
        result = predict_customer(form_data)
        recommendations = generate_recommendation(
            result["cleaned_record"], result["probability"], result["risk_level"]
        )
        return render_template(
            "single_result.html",
            customer=result["cleaned_record"],
            probability_pct=result["probability_pct"],
            prediction_label=result["prediction_label"],
            will_churn=result["will_churn"],
            risk_level=result["risk_level"],
            risk_css=result["risk_css"],
            signal_bars=result["signal_bars"],
            recommendations=recommendations,
            top_drivers=result["top_drivers"],
            model_name=result["model_name"],
            form_data=form_data,
            fields=FORM_FIELDS,
        )
    except PreprocessingError as exc:
        flash(str(exc), "error")
        return render_template("single_prediction.html", fields=FORM_FIELDS, form_data=form_data, errors=str(exc))
    except ModelLoadError as exc:
        flash(str(exc), "error")
        return render_template("single_prediction.html", fields=FORM_FIELDS, form_data=form_data, errors=str(exc))


@app.route("/bulk-prediction", methods=["GET", "POST"])
def bulk_prediction():
    if request.method == "GET":
        return render_template("bulk_prediction.html", model_ready=is_model_available())

    try:
        bundle = run_bulk_prediction(request.files.get("customer_csv"))
    except BulkPredictionError as exc:
        flash(str(exc), "error")
        return render_template("bulk_prediction.html", model_ready=is_model_available())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"retainiq_predictions_{timestamp}.csv"
    save_results_csv(bundle["results_df"], csv_filename)
    cache_path = save_bundle_cache(bundle, timestamp)

    session["latest_bundle_path"] = cache_path
    session["latest_results_filename"] = csv_filename
    session["latest_results_generated_at"] = datetime.now().strftime("%d %b %Y, %I:%M %p")

    kpis = compute_dashboard_kpis(bundle["results_df"])
    total_rows = len(bundle["results_df"])

    return render_template(
        "bulk_results.html",
        kpis=kpis,
        download_filename=csv_filename,
        records_by_prob=build_display_records(bundle, sort_by="probability")[:MAX_DISPLAY_ROWS],
        records_by_charges=build_display_records(bundle, sort_by="charges")[:MAX_DISPLAY_ROWS],
        records_by_tenure=build_display_records(bundle, sort_by="tenure")[:MAX_DISPLAY_ROWS],
        max_display_rows=MAX_DISPLAY_ROWS,
        total_rows=total_rows,
    )


@app.route("/download-results/<path:filename>")
def download_results(filename):
    safe_path = os.path.join(app.config["REPORTS_FOLDER"], os.path.basename(filename))
    if not os.path.exists(safe_path):
        abort(404)
    return send_file(safe_path, as_attachment=True, download_name=os.path.basename(filename))


@app.route("/dashboard")
def prediction_dashboard():
    bundle = _get_latest_bundle()
    if bundle is None:
        return render_template("prediction_dashboard.html", kpis=None, generated_at=None)

    kpis = compute_dashboard_kpis(bundle["results_df"])
    top_risk = build_top_risk_records(bundle)

    return render_template(
        "prediction_dashboard.html",
        kpis=kpis,
        top_risk=top_risk,
        generated_at=session.get("latest_results_generated_at"),
        download_filename=session.get("latest_results_filename"),
    )


@app.route("/powerbi-dashboard")
def powerbi_dashboard():
    return render_template("powerbi_dashboard.html", powerbi_url=POWERBI_URL)


@app.route("/reports")
def reports():
    report_files = []
    if os.path.isdir(app.config["REPORTS_FOLDER"]):
        for fname in sorted(os.listdir(app.config["REPORTS_FOLDER"]), reverse=True):
            if fname.lower().endswith(".csv"):
                fpath = os.path.join(app.config["REPORTS_FOLDER"], fname)
                try:
                    size_kb = round(os.path.getsize(fpath) / 1024, 1)
                    modified = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%d %b %Y, %I:%M %p")
                    row_count = max(sum(1 for _ in open(fpath, encoding="utf-8")) - 1, 0)
                    report_files.append({
                        "filename": fname, "size_kb": size_kb,
                        "modified": modified, "row_count": row_count,
                    })
                except OSError:
                    continue
    return render_template("reports.html", reports=report_files)


@app.route("/about")
def about():
    bundle = load_model()
    model_name = bundle.get("name") if bundle else None
    metrics = bundle.get("metrics") if bundle else None
    return render_template("about.html", model_name=model_name, metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
