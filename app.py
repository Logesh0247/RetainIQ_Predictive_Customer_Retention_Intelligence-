from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from utils.prediction import predict_customer
from utils.bulk_prediction import predict_bulk_customers

PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "templates" / "static"),
)

# ---------------------------------------
# Folders
# ---------------------------------------

UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_FOLDER)


# ---------------------------------------
# Home Page
# ---------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------
# Single Prediction
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    result = predict_customer(request.form)

    return render_template(
        "result.html",
        prediction=result["prediction"],
        probability=result["probability"],
        risk=result["risk"],
        recommendation=result["recommendation"],
    )


# ---------------------------------------
# Bulk Prediction Page
# ---------------------------------------

@app.route("/bulk-prediction")
def bulk_prediction():
    return render_template("bulk_prediction.html")


# ---------------------------------------
# Bulk Prediction
# ---------------------------------------

@app.route("/bulk-predict", methods=["POST"])
def bulk_predict():

    if "file" not in request.files:
        return render_template(
            "bulk_prediction.html",
            error="Please select a CSV file."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "bulk_prediction.html",
            error="Please select a CSV file."
        )

    upload_path = UPLOAD_FOLDER / file.filename

    file.save(upload_path)

    try:

        result = predict_bulk_customers(
            upload_path,
            OUTPUT_FOLDER
        )

        return render_template(

    "bulk_prediction.html",

    table=result["table"],

    top10=result["top10"],

    total=result["total"],

    churn=result["churn"],

    stay=result["stay"],

    high=result["high"],

    medium=result["medium"],

    low=result["low"],

    churn_rate=result["churn_rate"],

    retention_rate=result["retention_rate"],

    avg_monthly_charges=result["avg_monthly_charges"],
    
    revenue_at_risk=result["revenue_at_risk"],

    critical_customers=result["critical_customers"],

    avg_tenure=result["avg_tenure"],

    download_file=result["download_file"]

)

    except Exception as e:

        return render_template(

    "bulk_prediction.html",

    error=str(e),

    table=[],

    top10=[],

    total=0,

    churn=0,

    stay=0,

    high=0,

    medium=0,

    low=0,

    avg_probability=0,

    avg_monthly_charges=0,

    avg_tenure=0,

    download_file=""

)


# ---------------------------------------
# Download Result CSV
# ---------------------------------------

@app.route("/download/<filename>")
def download_file(filename):

    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename,
        as_attachment=True
    )


# ---------------------------------------
# Run App
# ---------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True
    )