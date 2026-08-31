"""
app.py
------
RetainIQ -- Customer Churn Prediction & Retention Intelligence Platform.
"""

import os
import re
import secrets
import logging
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, abort, session, jsonify, g
)
from werkzeug.middleware.proxy_fix import ProxyFix

from src.preprocessing import PreprocessingError, REQUIRED_COLUMNS
from utils.prediction import (
    predict_customer, is_model_available, ModelLoadError, load_model,
    MODEL_COMPARISON,
)
from utils.bulk_prediction import (
    run_bulk_prediction,
    run_sample_prediction,
    inspect_bulk_prediction_bytes,
    compute_dashboard_kpis,
    save_results_csv,
    build_display_records,
    build_top_risk_records,
    build_dashboard_page,
    build_customer_detail,
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
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

app.secret_key = os.environ.get(
    "RETAINIQ_SECRET_KEY",
    "retainiq-dev-secret-change-me"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SESSION_COOKIE_NAME"] = "retainiq_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_PATH"] = "/"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=400)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True

VISITOR_COOKIE = "retainiq_vid"
RUNS_COOKIE = "retainiq_runs"
VISITOR_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{16}$")

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
# Private per-visitor reports (this browser only — never a shared archive)
# ---------------------------------------------------------------------------

def _cookie_kwargs():
    # Preview is HTTPS inside an iframe: SameSite=None; Secure; Partitioned
    # (CHIPS) so the browser keeps the cookie. The test client uses http://.
    secure = not app.config.get("TESTING")
    return {
        "httponly": True,
        "samesite": "None" if secure else "Lax",
        "secure": secure,
        "path": "/",
        "max_age": int(timedelta(days=400).total_seconds()),
        "partitioned": secure,
    }


app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_PARTITIONED"] = True


@app.before_request
def _prepare_private_workspace():
    if app.config.get("TESTING"):
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
        app.config["SESSION_COOKIE_SECURE"] = False
        app.config["SESSION_COOKIE_PARTITIONED"] = False
    session.permanent = True
    vid = request.cookies.get(VISITOR_COOKIE) or session.get("visitor_id")
    if not vid or not VISITOR_ID_RE.match(str(vid)):
        vid = secrets.token_urlsafe(18)
    session["visitor_id"] = vid
    g.visitor_id = vid
    owned = list(session.get("user_runs") or [])
    for item in (request.cookies.get(RUNS_COOKIE) or "").split(","):
        item = item.strip()
        if RUN_ID_RE.match(item) and item not in owned:
            owned.append(item)
    session["user_runs"] = owned


@app.after_request
def _refresh_private_cookies(response):
    kwargs = _cookie_kwargs()
    vid = getattr(g, "visitor_id", None) or session.get("visitor_id")
    if vid and VISITOR_ID_RE.match(str(vid)):
        response.set_cookie(VISITOR_COOKIE, vid, **kwargs)
    owned = [item for item in (session.get("user_runs") or []) if RUN_ID_RE.match(str(item))]
    if owned:
        response.set_cookie(RUNS_COOKIE, ",".join(owned[-40:]), **kwargs)
    return response


def _runs_root():
    path = os.path.join(app.config["REPORTS_FOLDER"], "runs")
    os.makedirs(path, exist_ok=True)
    return path


def _run_folder(run_id, create=False):
    if not run_id or not RUN_ID_RE.match(str(run_id)):
        return None
    path = os.path.join(_runs_root(), run_id)
    if create:
        os.makedirs(path, exist_ok=True)
        return path
    return path if os.path.isdir(path) else None


def _csv_name(run_id):
    return f"retainiq_predictions_{run_id}.csv"


def _run_id_from_filename(filename):
    filename = os.path.basename(filename or "")
    prefix = "retainiq_predictions_"
    if filename.startswith(prefix) and filename.lower().endswith(".csv"):
        return filename[len(prefix):-4]
    return filename


def _owned_run_ids():
    return [item for item in (session.get("user_runs") or []) if RUN_ID_RE.match(str(item))]


def _remember_run(run_id, csv_filename, cache_path):
    session["latest_bundle_path"] = cache_path
    session["latest_results_filename"] = csv_filename
    session["latest_results_generated_at"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
    owned = _owned_run_ids()
    if run_id not in owned:
        owned.append(run_id)
    session["user_runs"] = owned
    session.modified = True


def _owns_report(filename):
    run_id = _run_id_from_filename(filename)
    return bool(run_id) and run_id in _owned_run_ids()


def _bundle_from_run(run_id):
    folder = _run_folder(run_id)
    if not folder:
        return None, None, None
    bundle = load_bundle_cache(os.path.join(folder, f".cache_{run_id}.pkl"))
    if bundle is None:
        return None, None, None
    return bundle, _csv_name(run_id), run_id


def _load_run_bundle(run=None):
    """Load a scored run. ?run= is enough — cookies are not required to view results."""
    run = (run or request.args.get("run") or "").strip()
    if run:
        return _bundle_from_run(run)

    latest = session.get("latest_results_filename")
    if latest:
        bundle, name, run_id = _bundle_from_run(_run_id_from_filename(latest))
        if bundle is not None:
            return bundle, name, run_id

    owned = _owned_run_ids()
    if owned:
        return _bundle_from_run(owned[-1])
    return None, None, None


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Product command center with only real model and prediction-session state."""
    model_bundle = load_model()
    prediction_bundle, _, run_id = _load_run_bundle()
    snapshot = None
    latest_activity = None
    if prediction_bundle is not None:
        snapshot = compute_dashboard_kpis(prediction_bundle["results_df"])
        generated_at = session.get("latest_results_generated_at")
        if not generated_at and run_id:
            folder = _run_folder(run_id)
            if folder:
                generated_at = datetime.fromtimestamp(os.path.getmtime(folder)).strftime(
                    "%d %b %Y, %I:%M %p"
                )
        latest_activity = {
            "label": "Bulk prediction completed",
            "detail": f"{len(prediction_bundle['results_df']):,} customers scored",
            "time": generated_at,
            "run_id": run_id,
        }
    return render_template(
        "home.html",
        model_ready=model_bundle is not None,
        model_name=model_bundle.get("name") if model_bundle else None,
        explainability_ready=bool(model_bundle and hasattr(model_bundle.get("model"), "coef_")),
        snapshot=snapshot,
        latest_activity=latest_activity,
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(8)
    csv_filename = _csv_name(run_id)
    folder = _run_folder(run_id, create=True)
    save_results_csv(bundle["results_df"], csv_filename, directory=folder)
    cache_path = save_bundle_cache(bundle, run_id, directory=folder)
    _remember_run(run_id, csv_filename, cache_path)
    return redirect(url_for("bulk_results_view", run=run_id))


@app.route("/bulk-results")
def bulk_results_view():
    try:
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
            top_risk=build_top_risk_records(bundle),
            max_display_rows=MAX_DISPLAY_ROWS,
            total_rows=total_rows,
            run_id=run_id,
        )
    except Exception as exc:
        logger.exception("Error displaying bulk results")
        flash(f"Could not render prediction results: {exc}", "error")
        return redirect(url_for("bulk_prediction"))


@app.route("/bulk-prediction", methods=["GET", "POST"])
def bulk_prediction():

    if request.method == "GET":
        return render_template(
            "bulk_prediction.html",
            model_ready=is_model_available(),
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            required_columns=REQUIRED_COLUMNS,
        )

    try:
        file_obj = request.files.get("customer_csv")
        bundle = run_bulk_prediction(file_obj)
        return _render_bulk_results(bundle)
    except BulkPredictionError as exc:
        flash(str(exc), "error")
        return render_template(
            "bulk_prediction.html", model_ready=is_model_available(),
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            required_columns=REQUIRED_COLUMNS,
        )
    except Exception:
        logger.exception("Unexpected error during bulk prediction")
        flash("Prediction could not be completed. Please verify the dataset and try again.", "error")
        return render_template(
            "bulk_prediction.html", model_ready=is_model_available(),
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            required_columns=REQUIRED_COLUMNS,
        )


@app.route("/bulk-prediction/validate", methods=["POST"])
def validate_bulk_prediction():
    """Validate and profile a CSV before the user starts model scoring."""
    file_obj = request.files.get("customer_csv")
    if file_obj is None or not getattr(file_obj, "filename", ""):
        return jsonify({"error": "Please select a CSV file before continuing."}), 400
    try:
        raw = file_obj.read()
        profile = inspect_bulk_prediction_bytes(raw, file_obj.filename)
        status = 200 if profile["valid"] else 422
        return jsonify(profile), status
    except BulkPredictionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Dataset validation failed")
        return jsonify({"error": "Dataset validation could not be completed. Check the CSV and try again."}), 500


@app.route("/bulk-prediction/sample", methods=["POST"])
def bulk_prediction_sample():
    try:
        bundle = run_sample_prediction()
        return _render_bulk_results(bundle)
    except BulkPredictionError as exc:
        flash(str(exc), "error")
        return render_template(
            "bulk_prediction.html", model_ready=is_model_available(),
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            required_columns=REQUIRED_COLUMNS,
        )
    except Exception:
        logger.exception("Unexpected error during sample bulk prediction")
        flash("The sample prediction could not be completed. Please try again.", "error")
        return render_template(
            "bulk_prediction.html", model_ready=is_model_available(),
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            required_columns=REQUIRED_COLUMNS,
        )


# ---------------------------------------------------------------------------
# Download prediction report
# ---------------------------------------------------------------------------

@app.route("/download-results/<path:filename>")
def download_results(filename):
    filename = os.path.basename(filename)
    run_id = (request.args.get("run") or "").strip() or _run_id_from_filename(filename)
    folder = _run_folder(run_id)
    if not folder:
        abort(404)

    safe_path = os.path.join(folder, filename)
    if not os.path.isfile(safe_path):
        csvs = [name for name in os.listdir(folder) if name.lower().endswith(".csv")]
        if not csvs:
            abort(404)
        filename = csvs[0]
        safe_path = os.path.join(folder, filename)

    return send_file(
        safe_path,
        as_attachment=True,
        download_name=filename
    )


# ---------------------------------------------------------------------------
# Latest Prediction Dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def prediction_dashboard():
    try:
        bundle, download_filename, run_id = _load_run_bundle()

        if bundle is None:
            return render_template(
                "prediction_dashboard.html",
                kpis=None,
                generated_at=None,
                model_ready=is_model_available(),
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
            dashboard_data=build_dashboard_page(bundle),
            generated_at=generated_at,
            download_filename=download_filename,
            run_id=run_id,
            model_ready=is_model_available(),
        )
    except Exception:
        logger.exception("Error loading prediction dashboard")
        flash("The prediction dashboard could not be loaded. Please run the prediction again.", "error")
        return redirect(url_for("bulk_prediction"))


@app.route("/dashboard/data")
def prediction_dashboard_data():
    """Filtered dashboard data and exact customer lookup for the interactive UI."""
    try:
        bundle, _, _ = _load_run_bundle(request.args.get("run"))
        if bundle is None:
            return jsonify({"error": "Prediction data is not available."}), 404
        customer_id = (request.args.get("customer_id") or "").strip()
        if customer_id:
            customer = build_customer_detail(bundle, customer_id)
            if customer is None:
                return jsonify({"error": "Customer not found. Please check the Customer ID."}), 404
            return jsonify({"customer": customer})
        return jsonify(build_dashboard_page(
            bundle,
            risk=request.args.get("risk", "All"),
            prediction=request.args.get("prediction", "All"),
            contract=request.args.get("contract", "All"),
            search=(request.args.get("search") or "").strip(),
            page=request.args.get("page", 1, type=int) or 1,
        ))
    except Exception:
        logger.exception("Dashboard data request failed")
        return jsonify({"error": "Dashboard data could not be loaded. Please try again."}), 500


# ---------------------------------------------------------------------------
# Historical Prediction Dashboard
#
# This route is used by the Review button on the Reports page.
# It loads the exact prediction bundle belonging to that report.
# ---------------------------------------------------------------------------

@app.route("/dashboard/review/<path:filename>")
def review_dashboard(filename):
    try:
        filename = os.path.basename(filename)
        run_id = _run_id_from_filename(filename)
        bundle, filename, run_id = _bundle_from_run(run_id)

        if bundle is None:
            flash("The dashboard data for this report is no longer available.", "error")
            return redirect(url_for("reports"))

        kpis = compute_dashboard_kpis(bundle["results_df"])
        top_risk = build_top_risk_records(bundle)

        folder = _run_folder(run_id)
        report_path = os.path.join(folder, filename) if folder else None

        generated_at = None
        if report_path and os.path.exists(report_path):
            generated_at = datetime.fromtimestamp(
                os.path.getmtime(report_path)
            ).strftime("%d %b %Y, %I:%M %p")

        return render_template(
            "prediction_dashboard.html",
            kpis=kpis,
            top_risk=top_risk,
            dashboard_data=build_dashboard_page(bundle),
            generated_at=generated_at,
            download_filename=filename,
            run_id=run_id,
            historical_dashboard=True,
            model_ready=is_model_available(),
        )
    except Exception:
        logger.exception("Error loading historical review dashboard")
        flash("This prediction dashboard could not be loaded.", "error")
        return redirect(url_for("reports"))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.route("/reports")
def reports():
    report_files = []
    for run_id in reversed(_owned_run_ids()):
        folder = _run_folder(run_id)
        if not folder:
            continue
        fname = _csv_name(run_id)
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            csvs = [name for name in os.listdir(folder) if name.lower().endswith(".csv")]
            if not csvs:
                continue
            fname = csvs[0]
            fpath = os.path.join(folder, fname)
        try:
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            modified = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime(
                "%d %b %Y, %I:%M %p"
            )
            with open(fpath, encoding="utf-8") as f:
                row_count = max(sum(1 for _ in f) - 1, 0)
            cache_path = os.path.join(folder, f".cache_{run_id}.pkl")
            report_files.append({
                "filename": fname,
                "size_kb": size_kb,
                "modified": modified,
                "row_count": row_count,
                "dashboard_available": os.path.exists(cache_path),
                "run_id": run_id,
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