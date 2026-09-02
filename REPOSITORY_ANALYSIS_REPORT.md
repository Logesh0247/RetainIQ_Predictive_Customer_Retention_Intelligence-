# RetainIQ — Full Repository Analysis Report

**Repository:** `Logesh0247/RetainIQ_Predictive_Customer_Retention_Intelligence-`  
**Analyzed commit:** `58177b0` (`files updated`, 2026-08-20)  
**Scope:** Entire tree (~145 files, ~96 MB of models, notebooks, Flask app, SQL, Power BI)

---

## 1. Executive summary

RetainIQ is an end-to-end **telecom customer-churn** project: IBM Telco dataset → cleaning / EDA / feature engineering notebooks → four trained classifiers → SHAP → rule-based retention recommendations → Power BI dashboard → a polished **Flask “Signal Ops Console”** for single/bulk scoring.

The **product story and UI are strong**. The **ML + production wiring is weaker than the README and About page claim**. The most important findings:

| Severity | Finding |
|----------|---------|
| Critical | Deployed “best” model is Random Forest, which **lost** the notebook bake-off (lowest F1). About page quotes **93.19% accuracy / 87.10% F1** — notebook outputs show **~79% / 0.57 F1**. |
| Critical | `src/preprocessing.py` encodes `gender` and `SeniorCitizen` incorrectly, and maps `PhoneService` to the wrong feature name. Production scores can be systematically wrong. |
| Critical | Linux case-sensitivity: Flask renders `single_prediction.html` but the file is `Single_prediction.html`. Single-prediction page will 500 on Linux/Docker. |
| High | README describes a **Streamlit** app and a directory layout that **does not exist**. FastAPI API imports a module that **does not exist**. |
| High | Hard-coded Windows `C:\Users\Hey!\...` paths in notebooks, tests, and SQL extract scripts. Repo is not portable. |
| Medium | No real tests, unpinned dependencies, secret key default, `debug=True`, pickle caches, huge generated CSVs committed. |

**Verdict:** Solid student/portfolio *concept* with a well-designed Flask UI. Not production-ready as documented. Fix encoding + model selection + docs before treating scores as decision-grade.

---

## 2. What the project is trying to do

Business problem (telecom): acquire customers cheaper than replacing them. The platform aims to answer:

1. **Who** is likely to churn? (probability + High/Medium/Low risk)
2. **Why?** (SHAP drivers)
3. **What to do?** (rule-based retention offers)
4. **What is the $ impact?** (at-risk monthly revenue, Power BI)

Dataset: **IBM Telco Customer Churn** (~7,043 rows). Target is binary `Churn Value` / `Churn Label`. Features cover demographics, services, contract, billing. Leakage columns (`Churn Score`, `Churn Reason`, `CLTV`, geo IDs) are dropped in the feature-engineering notebook.

---

## 3. Repository map (actual vs documented)

```
RetainIQ/
├── app.py                      Flask web app (real entry point)
├── api/app.py                  FastAPI stub (broken)
├── src/preprocessing.py        Feature adapter for the RF model
├── utils/                      predict, bulk predict, recommendations
├── templates/ + static/        Signal Ops Console (no JS)
├── models/                     ~96 MB pickled models + SHAP
├── notebooks/                  01–09 analysis pipeline
├── Data/                       raw + processed + sample CSV
├── Risk_segmentation/          notebook outputs + charts
├── retention_Intelligence/     recommendation outputs + charts
├── Power_BI_dashboard/         .pbix + master CSV + screenshots
├── Visualisations/             EDA + SHAP PNGs
├── sql/ + data_extraction/     SQL Server exploration / extract
├── schemas/request_schema.py   unused Pydantic model
├── tests/                      not tests (print scripts)
├── DockerFile, Procfile, runtime.txt
└── README.md                   out of date vs the code
```

README claims Streamlit, `app/app.py`, `src/preprocessing/`, LICENSE, docs/, images/, `preprocessor.pkl`. **None of that matches the tree.** The live app is Flask at repo-root `app.py`.

---

## 4. Architecture (as implemented)

```
CSV / HTML form
      │
      ▼
src.preprocessing  ──► 30 one-hot + numeric columns
      │
      ▼
models/best_model.pkl   (RandomForestClassifier, 200 trees)
      │
      ├── utils.prediction     single score + SHAP top drivers
      ├── utils.bulk_prediction  CSV ≤250k rows, KPIs, pickle cache
      └── utils.recommendation  if/else business rules
      │
      ▼
Flask pages: Home, Single, Bulk, Dashboard, Reports, About
      +
Power BI (offline .pbix on a joined master dataset)
```

There is **no** live CRM, no retraining job, no model registry, no auth.

---

## 5. Machine learning pipeline

### 5.1 Notebooks

| # | Notebook | Role | Notes |
|---|----------|------|--------|
| 01 | Data cleaning | Load raw Telco, inspect, clean | Hard-coded Windows path |
| 02 | EDA | Charts of contract, tenure, charges, payment, etc. | Outputs also in `Visualisations/EDA_images/` |
| 03 | Feature engineering | Drop leakage/geo; one-hot encode | Path even points at a **wrong** folder (`...\Platform\processed_data\...` missing `Data\`) |
| 04 | Train/test split | Stratified-style split artifacts | 5,634 train / 1,409 test, 30 features |
| 05 | Model training | LR, RF, XGB, LGBM | **Key results below** |
| 06 | SHAP | TreeExplainer on best_model | 25 MB `shap_explainer.pkl` |
| 07 | Risk segmentation | Probability → High/Med/Low | Outputs ~1.5k high / 0.3k med / 5.2k low |
| 08 | Retention engine | Rule-based offers + “revenue saved” | Heuristic, not causal |
| 09 | Power BI dataset | Join main + risk + retention | 8 cells, no markdown |

Notebooks are exploratory, not parameterized pipelines. They cannot be re-run on another machine without rewriting every path.

### 5.2 Bake-off results (from saved notebook outputs)

| Model | Accuracy | F1 |
|-------|----------|-----|
| **Logistic Regression** | **0.803** | **0.606** |
| LightGBM | 0.801 | 0.596 |
| XGBoost | 0.791 | 0.585 |
| Random Forest (n=200) | 0.790 | 0.569 |

LR also reported **ROC-AUC ≈ 0.849**, precision 0.65, recall 0.57 on the churn class (374 positives / 1,409). Class imbalance is real (~26.5% churn).

**The notebook then dumps Random Forest to `best_model.pkl` anyway.** That is the model the Flask app loads.

### 5.3 About-page vs evidence

`templates/about.html` states RF **Accuracy 93.19% / F1 87.10%**. Nothing in `05.Model_training.ipynb` supports those numbers. They look like a different run, a train-set score, or a documentation error. Treat them as **incorrect**.

### 5.4 Feature set (30 columns)

Numeric: `Tenure Months`, `Monthly Charges`, `Total Charges`  
Dummies: gender, senior, partner, dependents, phone/internet add-ons, contract, paperless, payment method.

README mentions engineered fields (`Revenue Per Month`, `Customer Value`, `Service Count`). Those exist in the **schema file** and marketing copy, **not** in the 30 columns the deployed model uses.

---

## 6. Web application (the strongest part)

`app.py` is a ~700-line Flask app with a coherent ops-console UX:

- **Home** — model online/offline, capability cards  
- **Single prediction** — 3-step CSS wizard (profile → services → billing)  
- **Bulk prediction** — CSV upload, 250 MB cap (configurable via `RETAINIQ_MAX_UPLOAD_MB`), 250k row cap (configurable via `RETAINIQ_MAX_ROWS`)  
- **Dashboard** — KPIs, risk mix, top-10 with SHAP + recs (session + pickle cache)  
- **Reports** — historical CSVs + “review” if `.cache_*.pkl` exists  
- **About** — product narrative  

UI craft is well above typical student projects:

- Design system in `static/style.css` (~540 lines): night/day theme, signal-strength meters, wizard, tabs, `:target` modals — **zero JavaScript**
- Accessible-ish focus rings, reduced-motion, mobile hamburger via checkbox hack
- Clear error pages (404 / 413 / 500)

App-layer hygiene that is **good**:

- Custom exceptions (`PreprocessingError`, `ModelLoadError`, `BulkPredictionError`)
- Download path uses `os.path.basename` (path traversal blocked)
- Model cached in memory after first load
- Flash messages + validation for empty/bad CSVs

App-layer hygiene that is **not good**:

- Default `RETAINIQ_SECRET_KEY = "retainiq-dev-secret-change-me"`
- `app.run(debug=True, host="0.0.0.0")` if started as `__main__`
- Session stores only a cache path; caches are world-readable pickles on disk
- Reports page counts CSV rows by reading every file on each request
- No auth — anyone who can hit the host can score and download reports

---

## 7. Critical code defects

### 7.1 Preprocessing encoder bugs (`src/preprocessing.py`)

`build_feature_frame` uses a helper that only matches the string `"Yes"`:

```python
def yes(col, feature):
    data.loc[df[col].eq("Yes"), feature] = 1

yes("gender", "Gender_Male")                 # gender is Male/Female, never "Yes"
yes("SeniorCitizen", "Senior Citizen_Yes")   # SeniorCitizen is 0/1 int
for c in ["Partner", "Dependents", "PhoneService"]:
    yes(c, f"{c}_Yes")                       # PhoneService_Yes ≠ "Phone Service_Yes"
```

Consequences:

- **Gender_Male is always 0**
- **Senior Citizen_Yes is always 0**
- **Phone Service_Yes is always 0** (wrong column name)

Partner / Dependents and the explicit mapping table (fiber, contract, e-check, etc.) still work. Predictions are not random, but they are **not** the same feature vector the model was trained on for three important fields.

### 7.2 Template case mismatch

| Flask asks for | File on disk |
|----------------|--------------|
| `single_prediction.html` | `templates/Single_prediction.html` |

Works on Windows (case-insensitive). **Fails on Linux, Docker (`python:3.13-slim`), and typical PaaS.**

### 7.3 Broken FastAPI service

`api/app.py` does `from src.predictor import ChurnPredictor`. There is **no** `src/predictor.py`. `schemas/request_schema.py` is unused and describes a different (underscore) feature schema plus leakage-like fields (`Churn_Score`, `CLTV`).

### 7.4 Deployment config contradictions

| File | Says |
|------|------|
| `runtime.txt` | Python **3.11.10** (Heroku-style) |
| `DockerFile` | `python:3.13-slim`, bind **8000**, `gunicorn app:app` |
| `app.py` | Flask debug server on **5000** |
| `Procfile` | `web: gunicorn app:app` |
| `requirements.txt` | unpinned; **no** xgboost, lightgbm, fastapi, pydantic, sqlalchemy, matplotlib, seaborn |

Sklearn / pickle models trained on one machine often fail to unpickle on 3.13 + a newer sklearn. That is a real deploy risk.

---

## 8. Data, SQL, and BI

**Data**

- Raw Telco CSV (~1.7 MB) plus cleaned / feature-engineered copies
- Train/test already materialized
- `Data/sample_customers.csv` for bulk demo
- **14 nearly identical 7k-row prediction CSVs** under `reports/` (~18 MB) — generated artifacts, should not live in git
- Risk + retention output CSVs duplicate the same 7k customers again

**SQL / extraction**

- Simple T-SQL: counts, churn rate, contract breakdown, 3 views, 3 indexes
- Typos in filenames (`Exoloration`, `Bussiness`) and inconsistent table names (`Customer_churn_Data` vs `Customer_Churn_Data`)
- Extract scripts require local SQL Server + Windows trusted connection + hard-coded paths
- `sqlalchemy` / `pyodbc` are not in `requirements.txt`

**Power BI**

- `Customer_Churn_prediction.pbix` + `.pbit` + 4 page screenshots
- Master dataset joins cleaned customers + risk + retention
- Useful for a portfolio walkthrough; not wired into the Flask app (CSS even has a Power BI iframe class that is unused)

---

## 9. Testing, quality, and repo hygiene

**Tests:** `tests/test_env.py` prints env vars. `tests/test_feature_columns.py` hard-codes a OneDrive path and prints columns. **Zero assertions. CI would not catch the encoding bugs.**

**Git hygiene issues**

- Single commit on `main` (“files updated”) — no history of the analysis journey
- Committed: `__pycache__/`, `.vs/` (Visual Studio), `desktop.ini`, 36 MB RF twice (`best_model.pkl` == `random_forest.pkl`), 25 MB SHAP pickle
- `.gitignore` is thin (no `models/*.pkl`, no `reports/*.csv`, no `.vs`, no notebooks checkpoints)
- No LICENSE file despite README saying MIT
- Clone URL in README is malformed (`git clone https:https://github.com/...`) and points at a different username (`Logesh-Data-Scientist` vs `Logesh0247`)

**Security**

- Pickle model + pickle report caches = arbitrary-code-execution if files are swapped
- No upload content-type beyond extension check
- Debug mode + default secret

**Logging**

- `logs/prediction.log` is leftover from an older predictor (repeated “Prediction=1, Probability=0.5450”). Current `app.py` uses `logging.getLogger("retainiq")` to stdout, not that file.

---

## 10. Documentation vs reality

| README / About claim | Reality |
|----------------------|---------|
| Streamlit web app | Flask + Jinja |
| Directory `app/`, `src/preprocessing/`, `docs/` | Flat `app.py`, single `src/preprocessing.py` |
| Selected model is best by metrics | RF is **worst** F1 of the four |
| RF 93% / 87% F1 | Notebook: ~79% / 0.57 F1 |
| Feature engineering: revenue/value/service count | Not in deployed 30 features |
| Real-time API listed as future *and* present | FastAPI file exists but cannot import |
| `uv pip install` + `streamlit run app/app.py` | Would fail as written |
| Production-ready / MLOps | No tests, no monitoring, no versioned training script |

The README is a long business-case essay (useful for interviews) but it is **not an accurate operator’s guide**.

---

## 11. What is genuinely good

1. **Complete narrative** from business question → EDA → models → explainability → segmentation → recommendations → BI → web UI.
2. **UI/UX**: distinctive “signal ops” metaphor, CSS-only interactivity, mobile layout, risk meters. Rare in student ML repos.
3. **Application structure** of the Flask layer: adapters, error types, bulk limits, report cache/review, download safety.
4. **Recommendation engine** is rule-based but *personalized* (contract, fiber + high bill, e-check, tenure, paperless). Better than a single canned sentence.
5. **Leakage awareness** in notebooks (drop Churn Score / Reason / CLTV from features).
6. **Visual artifacts** (EDA, SHAP, risk, retention, Power BI screenshots) make the project demoable.

---

## 12. Priority recommendations

### P0 — correctness (do first)

1. Fix `build_feature_frame`:
   - `Gender_Male` when `gender == "Male"`
   - `Senior Citizen_Yes` when `SeniorCitizen == 1`
   - `Phone Service_Yes` when `PhoneService == "Yes"`
2. Rename `templates/Single_prediction.html` → `single_prediction.html`.
3. Stop claiming 93% / 87% F1. Either re-evaluate honestly or change `best_model.pkl` to the actual winner (LR or a properly tuned booster).
4. Add a **unit test** that a known Male / senior / phone-service row produces the expected 30-vector, and that `predict_customer` returns a probability.

### P1 — reproducibility

5. Replace every `C:\Users\Hey!\...` path with `pathlib` relative to the repo root.
6. Pin `requirements.txt` (and include what notebooks/API actually need). Align Docker Python with `runtime.txt` (prefer 3.11).
7. Extract training from the notebook into `src/train.py` that writes metrics JSON the About page can read.
8. Delete or git-ignore duplicate `reports/*.csv`, `__pycache__`, `.vs`, and one of the identical 36 MB RF pickles.

### P2 — product / engineering

9. Either finish FastAPI (`src/predictor.py` wrapping `utils.prediction`) or delete `api/` and the unused schema.
10. Rewrite README to match Flask, real commands (`gunicorn app:app` / `python app.py`), real metrics, real tree.
11. `SECRET_KEY` required in prod; `debug=False`; don’t pickle report bundles (use parquet/JSON).
12. Consider class imbalance (class_weight / threshold tuning). Current RF F1 0.57 means many missed churners — the business metric that matters.

### P3 — stretch

13. Calibration plot + cost-sensitive threshold (retention offer cost vs CLTV).
14. Don’t use SHAP on every bulk row (already limited to top 10 — good). Persist explanations.
15. Add auth if this is ever exposed beyond localhost.

---

## 13. Bottom line

RetainIQ is a **complete portfolio platform**: the business framing, EDA, Power BI, and especially the Flask console are impressive. The scientific and engineering core has three integrity problems — **wrong “best” model, inflated metrics in the UI, and broken feature encoding** — plus a Linux-breaking template name and a README that describes a different project.

Fix those, and this is a credible end-to-end churn product demo. Until then, treat every probability the app prints as **untrustworthy**.
