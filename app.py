"""
app.py
------
RetainIQ -- Customer Churn Prediction & Retention Intelligence Platform.
"""

import os
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, abort, session, jsonify
)

from src.preprocessing import PreprocessingError
from utils.prediction import (
    predict_customer, is_model_available, ModelLoadError, load_model,
    MODEL_COMPARISON,
)
from utils.bulk_prediction import (
    run_bulk_prediction,
    run_sample_prediction,
    compute_dashboard_kpis,
    save_results_csv,
    build_display_records,
    build_top_risk_records,
    save_bundle_cache,
    load_bundle_cache,
    BulkPredictionError,
    MAX_DISPLAY_ROWS,
    REPORTS_DIR,
)
from utils.recommendation import generate_recommendation


# ---------------------------------------------------------------------------
# Single prediction form fields
# ---------------------------------------------------------------------------

FORM_FIELDS = [
    {
        "name": "customerID",
        "label": "Customer ID",
        "type": "text",
        "required": False,
        "placeholder": "e.g. 7590-VHVEG",
    },
    {
        "name": "gender",
        "label": "Gender",
        "type": "select",
        "options": ["Female", "Male"],
        "required": True,
    },
    {
        "name": "SeniorCitizen",
        "label": "Senior Citizen",
        "type": "select",
        "options": ["0", "1"],
        "required": True,
    },
    {
        "name": "Partner",
        "label": "Partner",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
    },
    {
        "name": "Dependents",
        "label": "Dependents",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
    },
    {
        "name": "tenure",
        "label": "Tenure (Months)",
        "type": "number",
        "required": True,
        "min": 0,
        "max": 100,
    },
    {
        "name": "PhoneService",
        "label": "Phone Service",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
    },
    {
        "name": "MultipleLines",
        "label": "Multiple Lines",
        "type": "select",
        "options": ["Yes", "No", "No phone service"],
        "required": True,
    },
    {
        "name": "InternetService",
        "label": "Internet Service",
        "type": "select",
        "options": ["DSL", "Fiber optic", "No"],
        "required": True,
    },
    {
        "name": "OnlineSecurity",
        "label": "Online Security",
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "required": True,
    },
    {
        "name": "OnlineBackup",
        "label": "Online Backup",
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "required": True,
    },
    {
        "name": "DeviceProtection",
        "label": "Device Protection",
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "required": True,
    },
    {
        "name": "TechSupport",
        "label": "Tech Support",
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "required": True,
    },
    {
        "name": "StreamingTV",
        "label": "Streaming TV",
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "required": True,
    },
    {
        "name": "StreamingMovies",
        "label": "Streaming Movies",
        "type": "select",
        "options": ["Yes", "No", "No internet service"],
        "required": True,
    },
    {
        "name": "Contract",
        "label": "Contract",
        "type": "select",
        "options": ["Month-to-month", "One year", "Two year"],
        "required": True,
    },
    {
        "name": "PaperlessBilling",
        "label": "Paperless Billing",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
    },
    {
        "name": "PaymentMethod",
        "label": "Payment Method",
        "type": "select",
        "options": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "required": True,
    },
    {
        "name": "MonthlyCharges",
        "label": "Monthly Charges",
        "type": "number",
        "required": True,
        "step": "0.01",
        "min": 0,
    },
    {
        "name": "TotalCharges",
        "label": "Total Charges",
        "type": "number",
        "required": False,
        "step": "0.01",
        "min": 0,
    },
]


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

app.secret_key = os.environ.get(
    "RETAINIQ_SECRET_KEY",
    "retainiq-dev-secret-change-me"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["REPORTS_FOLDER"] = REPORTS_DIR

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)

os.makedirs(
    app.config["REPORTS_FOLDER"],
    exist_ok=True
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("retainiq")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template(
        "error.html",
        code=404,
        message="That signal doesn't exist. The page you're looking for isn't here."
    ), 404


@app.errorhandler(413)
def too_large(e):
    return render_template(
        "error.html",
        code=413,
        message="The uploaded file is too large (max 10 MB)."
    ), 413


@app.errorhandler(500)
def server_error(e):
    logger.exception("Unhandled server error")

    return render_template(
        "error.html",
        code=500,
        message="Something went wrong on our end. Please try again."
    ), 500


# ---------------------------------------------------------------------------
# Helper: latest bundle
# ---------------------------------------------------------------------------

def _get_latest_bundle():
    path = session.get("latest_bundle_path")
    return load_bundle_cache(path) if path else None


def _latest_run_file():
    return os.path.join(app.config["REPORTS_FOLDER"], ".latest_run")


def _remember_run(timestamp, csv_filename, cache_path):
    session["latest_bundle_path"] = cache_path
    session["latest_results_filename"] = csv_filename
    session["latest_results_generated_at"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
    try:
        with open(_latest_run_file(), "w", encoding="utf-8") as handle:
            handle.write(timestamp)
    except OSError:
        pass


def _newest_cache_path():
    folder = app.config["REPORTS_FOLDER"]
    if not os.path.isdir(folder):
        return None
    caches = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith(".cache_") and name.endswith(".pkl")
    ]
    if not caches:
        return None
    return max(caches, key=os.path.getmtime)


def _load_run_bundle(run=None):
    run = (run or request.args.get("run") or "").strip()
    if not run:
        try:
            with open(_latest_run_file(), encoding="utf-8") as handle:
                run = handle.read().strip()
        except OSError:
            run = ""

    if run and all(ch.isalnum() or ch == "_" for ch in run):
        path = os.path.join(app.config["REPORTS_FOLDER"], f".cache_{run}.pkl")
        bundle = load_bundle_cache(path)
        if bundle is not None:
            return bundle, f"retainiq_predictions_{run}.csv", run

    bundle = _get_latest_bundle()
    if bundle is not None:
        return bundle, session.get("latest_results_filename"), None

    path = _newest_cache_path()
    bundle = load_bundle_cache(path)
    if bundle is not None and path:
        stamp = os.path.basename(path)[len(".cache_"):-4]
        return bundle, f"retainiq_predictions_{stamp}.csv", stamp
    return None, None, None


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "home.html",
        model_ready=is_model_available()
    )


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

@app.route("/single-prediction", methods=["GET", "POST"])
def single_prediction():

    if request.method == "GET":

        return render_template(
            "single_prediction.html",
            fields=FORM_FIELDS,
            form_data={},
            errors=None
        )

    form_data = request.form.to_dict()

    try:

        result = predict_customer(form_data)

        recommendations = generate_recommendation(
            result["cleaned_record"],
            result["probability"],
            result["risk_level"]
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
            primary_action=recommendations[0] if recommendations else "Monitor account.",
            top_drivers=result["top_drivers"],
            model_name=result["model_name"],
            form_data=form_data,
            fields=FORM_FIELDS,
        )

    except PreprocessingError as exc:

        flash(str(exc), "error")

        return render_template(
            "single_prediction.html",
            fields=FORM_FIELDS,
            form_data=form_data,
            errors=str(exc)
        )

    except ModelLoadError as exc:

        flash(str(exc), "error")

        return render_template(
            "single_prediction.html",
            fields=FORM_FIELDS,
            form_data=form_data,
            errors=str(exc)
        )


# ---------------------------------------------------------------------------
# Bulk prediction
# ---------------------------------------------------------------------------

def _render_bulk_results(bundle):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"retainiq_predictions_{timestamp}.csv"
    save_results_csv(bundle["results_df"], csv_filename)
    cache_path = save_bundle_cache(bundle, timestamp)
    _remember_run(timestamp, csv_filename, cache_path)
    return redirect(url_for("bulk_results_view", run=timestamp))


@app.route("/bulk-results")
def bulk_results_view():
    bundle, download_filename, run_id = _load_run_bundle()
    if bundle is None:
        flash("Run a bulk prediction first — choose a CSV or use the sample file.", "error")
        return redirect(url_for("bulk_prediction"))

    kpis = compute_dashboard_kpis(bundle["results_df"])
    total_rows = len(bundle["results_df"])
    return render_template(
        "bulk_results.html",
        kpis=kpis,
        download_filename=download_filename,
        records_by_prob=build_display_records(bundle, sort_by="probability")[:MAX_DISPLAY_ROWS],
        records_by_charges=build_display_records(bundle, sort_by="charges")[:MAX_DISPLAY_ROWS],
        records_by_tenure=build_display_records(bundle, sort_by="tenure")[:MAX_DISPLAY_ROWS],
        max_display_rows=MAX_DISPLAY_ROWS,
        total_rows=total_rows,
        run_id=run_id,
    )


@app.route("/bulk-prediction", methods=["GET", "POST"])
def bulk_prediction():

    if request.method == "GET":
        return render_template("bulk_prediction.html", model_ready=is_model_available())

    try:
        bundle = run_bulk_prediction(request.files.get("customer_csv"))
    except BulkPredictionError as exc:
        flash(str(exc), "error")
        return render_template("bulk_prediction.html", model_ready=is_model_available())

    return _render_bulk_results(bundle)


@app.route("/bulk-prediction/sample", methods=["POST"])
def bulk_prediction_sample():
    try:
        bundle = run_sample_prediction()
    except BulkPredictionError as exc:
        flash(str(exc), "error")
        return render_template("bulk_prediction.html", model_ready=is_model_available())
    return _render_bulk_results(bundle)


# ---------------------------------------------------------------------------
# Download prediction report
# ---------------------------------------------------------------------------

@app.route("/download-results/<path:filename>")
def download_results(filename):

    safe_path = os.path.join(
        app.config["REPORTS_FOLDER"],
        os.path.basename(filename)
    )

    if not os.path.exists(safe_path):

        abort(404)

    return send_file(
        safe_path,
        as_attachment=True,
        download_name=os.path.basename(filename)
    )


# ---------------------------------------------------------------------------
# Latest Prediction Dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def prediction_dashboard():
    bundle, download_filename, run_id = _load_run_bundle()

    if bundle is None:
        return render_template(
            "prediction_dashboard.html",
            kpis=None,
            generated_at=None,
        )

    kpis = compute_dashboard_kpis(bundle["results_df"])
    top_risk = build_top_risk_records(bundle)
    generated_at = session.get("latest_results_generated_at")
    if not generated_at and run_id:
        generated_at = run_id.replace("_", " ")

    return render_template(
        "prediction_dashboard.html",
        kpis=kpis,
        top_risk=top_risk,
        generated_at=generated_at,
        download_filename=download_filename,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Historical Prediction Dashboard
#
# This route is used by the Review button on the Reports page.
# It loads the exact prediction bundle belonging to that report.
# ---------------------------------------------------------------------------

@app.route("/dashboard/review/<path:filename>")
def review_dashboard(filename):

    filename = os.path.basename(filename)

    # Only CSV reports can be reviewed.
    if not filename.lower().endswith(".csv"):

        abort(404)

    # Convert:
    #
    # retainiq_predictions_20260817_120000.csv
    #
    # into:
    #
    # .cache_20260817_120000.pkl
    #
    timestamp_part = filename[:-4]

    prefix = "retainiq_predictions_"

    if not timestamp_part.startswith(prefix):

        abort(404)

    timestamp = timestamp_part[len(prefix):]

    cache_filename = f".cache_{timestamp}.pkl"

    cache_path = os.path.join(
        app.config["REPORTS_FOLDER"],
        cache_filename
    )

    bundle = load_bundle_cache(cache_path)

    if bundle is None:

        flash("The dashboard data for this report is no longer available.","error")

        return redirect(url_for("reports"))

    kpis = compute_dashboard_kpis(bundle["results_df"])

    top_risk = build_top_risk_records(bundle)

    # Use the CSV file's actual modification time
    # so the dashboard identifies the historical run correctly.
    report_path = os.path.join(app.config["REPORTS_FOLDER"],filename)

    generated_at = None

    if os.path.exists(report_path):

        generated_at = datetime.fromtimestamp(
            os.path.getmtime(report_path)
        ).strftime("%d %b %Y, %I:%M %p")

    return render_template(
        "prediction_dashboard.html",
        kpis=kpis,
        top_risk=top_risk,
        generated_at=generated_at,
        download_filename=filename,
        historical_dashboard=True,
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.route("/reports")
def reports():

    report_files = []

    if os.path.isdir(
        app.config["REPORTS_FOLDER"]
    ):

        for fname in sorted(
            os.listdir(
                app.config["REPORTS_FOLDER"]
            ),
            reverse=True
        ):

            if not fname.lower().endswith(".csv"):

                continue

            fpath = os.path.join(
                app.config["REPORTS_FOLDER"],
                fname
            )

            try:

                size_kb = round(
                    os.path.getsize(fpath) / 1024,
                    1
                )

                modified = datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).strftime(
                    "%d %b %Y, %I:%M %p"
                )

                with open(
                    fpath,
                    encoding="utf-8"
                ) as f:

                    row_count = max(
                        sum(1 for _ in f) - 1,
                        0
                    )

                # Check whether the matching
                # prediction bundle exists.
                timestamp_part = fname[:-4]

                prefix = "retainiq_predictions_"

                dashboard_available = False

                if timestamp_part.startswith(prefix):

                    timestamp = timestamp_part[
                        len(prefix):
                    ]

                    cache_filename = (
                        f".cache_{timestamp}.pkl"
                    )

                    cache_path = os.path.join(
                        app.config["REPORTS_FOLDER"],
                        cache_filename
                    )

                    dashboard_available = os.path.exists(
                        cache_path
                    )

                report_files.append({
                    "filename": fname,
                    "size_kb": size_kb,
                    "modified": modified,
                    "row_count": row_count,
                    "dashboard_available": dashboard_available,
                })

            except OSError:

                continue

    return render_template(
        "reports.html",
        reports=report_files
    )


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

@app.route("/about")
def about():

    bundle = load_model()

    model_name = (
        bundle.get("name")
        if bundle
        else None
    )

    metrics = (
        bundle.get("metrics")
        if bundle
        else None
    )

    return render_template(
        "about.html",
        model_name=model_name,
        metrics=metrics,
        comparison=MODEL_COMPARISON,
    )


# ---------------------------------------------------------------------------
# JSON API (same model as the UI)
# ---------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "Running",
        "model_ready": is_model_available(),
        "model": (load_model() or {}).get("name"),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with customer fields."}), 400
    try:
        result = predict_customer(payload)
        recs = generate_recommendation(
            result["cleaned_record"],
            result["probability"],
            result["risk_level"],
        )
        return jsonify({
            "customer_id": str(result["cleaned_record"].get("customerID", "")),
            "probability": result["probability"],
            "probability_pct": result["probability_pct"],
            "prediction": result["prediction_label"],
            "risk_level": result["risk_level"],
            "top_drivers": result["top_drivers"],
            "primary_action": recs[0] if recs else "Monitor account.",
            "recommendations": recs,
            "model_name": result["model_name"],
        })
    except (PreprocessingError, ModelLoadError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("API prediction failed")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )