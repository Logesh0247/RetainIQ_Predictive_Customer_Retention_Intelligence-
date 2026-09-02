# RetainIQ — Predictive Customer Retention Intelligence Platform

<p align="center">

**An Explainable Machine Learning Platform for Customer Churn Prediction, Risk Intelligence, and Retention Strategy**

</p>

<p align="center">

[Live Application](https://retainiq-predictive-customer-retention-zq6x.onrender.com)

</p>

---

## 📌 Overview

**RetainIQ** is an end-to-end machine learning and business intelligence platform designed to help telecommunications businesses identify customers who are at risk of churn and support proactive customer retention decisions.

Instead of limiting churn prediction to a simple **Churn / No-Churn** classification, RetainIQ transforms customer data into actionable retention intelligence through:

* Customer churn prediction
* Churn probability estimation
* Customer risk segmentation
* Prediction explanations
* Feature contribution analysis
* Retention recommendations
* Bulk customer prediction
* Interactive prediction dashboards
* Revenue-at-risk analysis
* Business reports
* Power BI analytics
* Production web deployment

The platform combines **Data Science, Machine Learning, Explainable AI, Business Intelligence, and Web Application Development** into a single solution.

---

# 🎯 Project Objectives

The primary objectives of RetainIQ are to:

1. Predict customers who are likely to churn.
2. Estimate the probability of customer churn.
3. Identify important factors associated with customer churn.
4. Explain individual customer predictions.
5. Segment customers according to churn risk.
6. Identify customers requiring immediate retention attention.
7. Generate data-driven retention recommendations.
8. Estimate potential monthly revenue at risk.
9. Provide interactive dashboards for business users.
10. Enable single and bulk customer prediction.
11. Generate analytical reports.
12. Deploy the machine learning solution as a web application.

---

# 💼 Business Problem

Customer churn is a major challenge for telecommunications companies operating in highly competitive markets.

Customers can switch service providers because of factors such as:

* Pricing
* Contract type
* Service quality
* Lack of technical support
* Billing preferences
* Internet service
* Customer experience
* Limited service benefits

Traditional reporting systems are primarily descriptive. They explain **what happened**, but they do not necessarily identify **who is likely to churn next** or **what action should be taken**.

RetainIQ addresses this problem by transforming historical customer information into predictive and actionable intelligence.

### Key Business Questions

The platform is designed to answer four important questions:

### 1. Who is likely to churn?

Predict customers with a high probability of discontinuing their services.

### 2. Why are they likely to churn?

Identify the customer characteristics and factors contributing to the prediction.

### 3. What action should be taken?

Provide retention recommendations based on customer risk factors.

### 4. What business value is at risk?

Estimate the potential monthly revenue associated with customers identified as being at risk.

---

# 🔬 Project Scope

The project covers the complete machine learning lifecycle:

```text
Business Understanding
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Explainability
        ↓
Risk Segmentation
        ↓
Retention Intelligence
        ↓
Web Application
        ↓
Dashboard & Reports
        ↓
Deployment
```

---

# 📊 Dataset

The project uses the **IBM Telco Customer Churn Dataset**, containing customer demographic information, service subscriptions, contract details, billing information, and churn outcomes.

### Dataset Summary

| Property              | Value                 |
| --------------------- | --------------------- |
| Industry              | Telecommunications    |
| Problem Type          | Binary Classification |
| Original Records      | 7,043                 |
| Original Features     | 34                    |
| Final ML Features     | 30                    |
| Target Variable       | Churn Label           |
| Churned Customers     | 1,869                 |
| Non-Churned Customers | 5,174                 |
| Churn Rate            | 26.54%                |
| ML Approach           | Supervised Learning   |

### Churn Distribution

| Customer Status |     Count | Percentage |
| --------------- | --------: | ---------: |
| Stayed          |     5,174 |     73.46% |
| Churned         |     1,869 |     26.54% |
| **Total**       | **7,043** |   **100%** |

---

# 🧾 Dataset Features

The dataset contains information from several categories.

### Customer Information

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents

### Geographic Information

* Country
* State
* City
* Zip Code
* Latitude
* Longitude
* Lat Long

### Subscription Information

* Tenure Months
* Phone Service
* Multiple Lines
* Internet Service

### Service Features

* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Contract and Billing

* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

### Churn Information

* Churn Label
* Churn Value
* Churn Score
* Churn Reason

### Customer Value

* CLTV

---

# 🧹 Data Preprocessing

The preprocessing pipeline prepares the raw telecom dataset for machine learning.

The process includes:

* Data type correction
* Missing-value analysis
* Missing-value treatment
* Duplicate analysis
* Categorical value normalization
* Numerical feature processing
* Removal of identifier columns
* Removal of geographic fields not required for prediction
* Prevention of target/data leakage
* Categorical encoding
* Feature scaling
* Train-test splitting

### Leakage Prevention

Variables that directly describe churn outcomes or are generated after churn occurs are excluded from model training.

Examples include:

* Churn Label
* Churn Reason
* Churn Category
* Churn Score
* Customer Status
* Other post-outcome information

This prevents the model from receiving information that would not realistically be available before churn.

---

# ⚙️ Feature Engineering

The project transforms customer information into machine-learning-ready features.

The final model uses **30 predictive features**, including:

* Tenure Months
* Monthly Charges
* Total Charges
* Gender
* Senior Citizen
* Partner
* Dependents
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method

Categorical variables are transformed into numerical representations using encoding techniques.

---

# 📈 Exploratory Data Analysis

Extensive exploratory analysis was performed to understand customer behavior and identify potential churn patterns.

The analysis includes:

* Churn distribution
* Contract vs churn
* Tenure vs churn
* Monthly charges vs churn
* Payment method vs churn
* Internet service vs churn
* Senior citizen analysis
* Customer service analysis
* Customer behavior analysis
* Correlation analysis
* Distribution analysis

The EDA stage helps identify patterns and relationships before model development.

---

# 🤖 Machine Learning Models

Multiple supervised machine learning algorithms were developed and evaluated.

The project evaluates:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

The models were compared using multiple evaluation metrics rather than relying only on accuracy.

---

# 📊 Model Evaluation

The evaluation process includes:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

## Final Model Comparison

Based on the current production model artifacts and evaluation results:

| Model                   |   Accuracy |   F1 Score |
| ----------------------- | ---------: | ---------: |
| **Logistic Regression** | **80.34%** | **60.60%** |
| LightGBM                |     80.06% |     59.57% |
| XGBoost                 |     79.06% |     58.51% |
| Random Forest           |     78.99% |     56.85% |

### Selected Model

**Logistic Regression** is currently used as the production model because it achieved the strongest overall performance among the evaluated models, particularly in terms of accuracy and F1 score.

### Final Production Model Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **80.34%** |
| Precision | **64.74%** |
| Recall    | **56.95%** |
| F1 Score  | **60.60%** |
| ROC-AUC   | **84.91%** |

---

# 🔍 Explainable AI

A key objective of RetainIQ is to make machine learning predictions understandable to business users.

The platform provides prediction explanations by analyzing the contribution of customer features to the model's decision.

For the current production Logistic Regression model, feature contributions are derived from the model coefficients and customer feature values.

This allows users to understand:

* Which factors increase churn risk
* Which factors decrease churn risk
* Why a customer received a particular prediction
* Which customer characteristics require attention

### Explainability Workflow

```text
Customer Data
      ↓
Preprocessing
      ↓
ML Model
      ↓
Churn Probability
      ↓
Feature Contributions
      ↓
Prediction Explanation
```

> Earlier experimentation in the project also explored SHAP-based explainability. The production explanation layer is aligned with the currently deployed Logistic Regression model.

---

# 🚦 Customer Risk Segmentation

RetainIQ converts churn probability into actionable risk categories.

| Churn Probability | Risk Level  |
| ----------------- | ----------- |
| `< 30%`           | Low Risk    |
| `30% – < 60%`     | Medium Risk |
| `≥ 60%`           | High Risk   |

This allows business teams to prioritize customers instead of treating every customer equally.

### Risk Intelligence Workflow

```text
Churn Probability
        ↓
Risk Classification
        ↓
Low / Medium / High
        ↓
Customer Prioritization
        ↓
Retention Action
```

---

# 💡 Retention Intelligence

RetainIQ goes beyond prediction by providing actionable retention recommendations.

Recommendations are generated based on customer characteristics and identified risk factors.

Possible actions include:

* Personalized retention offers
* Contract upgrade incentives
* Technical support offers
* Online security offers
* Autopay recommendations
* Loyalty incentives
* Personalized retention calls
* Service improvement offers

The objective is to transform:

> **Prediction → Explanation → Action**

rather than stopping at prediction.

---

# 💰 Revenue-at-Risk Analysis

The platform provides business-oriented revenue analysis for customers identified as being at risk.

Instead of claiming that revenue has already been saved, RetainIQ uses the more appropriate concept:

### **Monthly Revenue at Risk**

This allows organizations to identify the approximate recurring monthly charges associated with customers who may churn.

This metric can support:

* Retention prioritization
* Campaign planning
* Revenue-risk analysis
* Executive decision-making

---

# 🌐 Web Application

RetainIQ is implemented as a **Flask-based web application** and deployed as a production web service.

### Main Application Modules

```text
Home
 │
 ├── Single Prediction
 │
 ├── Bulk Prediction
 │
 ├── Prediction Dashboard
 │
 ├── Reports
 │
 └── About
```

---

# 👤 Single Customer Prediction

The single prediction module allows users to enter customer information and receive:

* Churn prediction
* Churn probability
* Risk level
* Customer insights
* Prediction explanation
* Retention recommendation

### Workflow

```text
Customer Information
        ↓
Input Validation
        ↓
Preprocessing
        ↓
Logistic Regression
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
Explanation
        ↓
Retention Recommendation
```

---

# 📁 Bulk Customer Prediction

The bulk prediction module allows multiple customer records to be processed using an uploaded dataset.

The system:

1. Accepts customer data.
2. Validates the uploaded file.
3. Processes the customer records.
4. Generates churn predictions.
5. Calculates churn probabilities.
6. Assigns risk levels.
7. Generates retention insights.
8. Displays the results.
9. Allows results to be exported.

This makes the platform suitable for analyzing large groups of customers rather than only individual records.

---

# 📊 Prediction Dashboard

The Prediction Dashboard provides an operational view of customer churn risk.

It includes metrics and visualizations such as:

* Total customers
* Predicted churners
* Churn rate
* Average churn probability
* High-risk customers
* Medium-risk customers
* Low-risk customers
* Average monthly charges
* Average customer tenure
* Revenue at risk
* Highest-risk customers

The dashboard also supports customer-level analysis through search and filtering functionality.

---

# 📑 Reports

The platform provides report-oriented functionality for analyzing prediction results.

Reports can contain:

* Customer prediction results
* Risk distribution
* Churn statistics
* Revenue-at-risk information
* Customer insights
* Retention recommendations

Results can also be exported for further analysis.

---

# 📊 Power BI Dashboard

RetainIQ is complemented by a Microsoft Power BI dashboard for business intelligence and executive analysis.

The dashboard provides views related to:

* Executive KPIs
* Customer churn
* Customer revenue
* Churn risk
* Customer segmentation
* Geographic analysis
* Service analysis
* Retention strategy

The Power BI layer provides descriptive and diagnostic analytics, while the machine learning application provides predictive and prescriptive intelligence.

### Analytics Architecture

```text
Historical Customer Data
          ↓
     Power BI
          ↓
Descriptive Analytics

Historical Customer Data
          ↓
 Machine Learning
          ↓
Predictive Analytics
          ↓
Risk & Recommendations
          ↓
Prescriptive Analytics
```

---

# 🏗️ System Architecture

```text
                    Customer Dataset
                           │
                           ▼
                  Data Preprocessing
                           │
                           ▼
                Exploratory Data Analysis
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                  Machine Learning Models
                           │
                           ▼
                    Model Evaluation
                           │
                           ▼
                 Final Production Model
                  Logistic Regression
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Churn Probability          Explainability
              │                         │
              ▼                         ▼
       Risk Segmentation       Feature Contributions
              │                         │
              └────────────┬────────────┘
                           ▼
                Retention Intelligence
                           │
                ┌──────────┴───────────┐
                ▼                      ▼
        Flask Web Application      Power BI
                │                      │
                ▼                      ▼
       Dashboard / Reports      Executive Analytics
```

---

# 🛠️ Technology Stack

## Programming

* Python

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

## Explainable AI

* SHAP
* Logistic Regression feature contributions

## Model Persistence

* Joblib

## Web Application

* Flask
* HTML
* CSS
* JavaScript where required by the application

## Data Visualization

* Matplotlib
* Seaborn
* Plotly

## Business Intelligence

* Microsoft Power BI

## Deployment

* Render
* Gunicorn
* Docker configuration

## Version Control

* Git
* GitHub

---

# 📂 Project Structure

```text
RetainIQ_Signal_Ops_Console/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── Dockerfile
├── Procfile
│
├── app.py
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   ├── shap_explainer.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── lightbgm.pkl
│
├── utils/
│   └── prediction.py
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── ...
│
├── static/
│   ├── style.css
│   └── ...
│
├── src/
│   └── ...
│
├── notebooks/
│   ├── 01_Business_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   ├── 07_SHAP_Analysis.ipynb
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
│   └── ...
│
├── reports/
│   └── ...
│
├── tests/
│   └── ...
│
└── images/
    └── ...
```

> The exact contents of some directories may evolve as the project is maintained.

---

# 🔄 Complete Project Workflow

### Phase 1 — Business Understanding

* Identify the customer churn problem
* Define business objectives
* Identify analytical requirements

### Phase 2 — Data Collection

* Acquire telecom customer churn dataset
* Understand available customer attributes

### Phase 3 — Data Cleaning

* Handle missing values
* Correct data types
* Detect duplicates
* Normalize categorical values
* Remove unnecessary fields

### Phase 4 — Exploratory Data Analysis

* Analyze churn distribution
* Study customer characteristics
* Identify relationships between variables
* Analyze service and contract behavior

### Phase 5 — Feature Engineering

* Select relevant variables
* Encode categorical variables
* Prepare numerical features
* Create the final machine learning feature matrix

### Phase 6 — Model Development

Train and evaluate:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

### Phase 7 — Model Evaluation

Compare models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### Phase 8 — Final Model Selection

Select Logistic Regression as the current production model based on evaluation performance.

### Phase 9 — Explainability

Generate feature-level explanations for predictions.

### Phase 10 — Risk Segmentation

Convert churn probabilities into:

* Low Risk
* Medium Risk
* High Risk

### Phase 11 — Retention Intelligence

Generate customer-specific retention recommendations.

### Phase 12 — Business Intelligence

Provide Power BI dashboards and operational prediction dashboards.

### Phase 13 — Web Application

Integrate the complete prediction pipeline into Flask.

### Phase 14 — Deployment

Deploy the application using a production WSGI server and cloud hosting infrastructure.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Logesh-Data-Scientist/RetainIQ_Predictive_Customer_Retention_Intelligence-.git
```

## 2. Navigate to the Project

```bash
cd RetainIQ_Predictive_Customer_Retention_Intelligence-
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Application

```bash
python app.py
```

The application will be available locally through the Flask development server.

---

# 🌍 Live Deployment

The current deployed application is available at:

**https://retainiq-predictive-customer-retention-zq6x.onrender.com**

The production deployment uses:

* Flask
* Gunicorn
* Render
* Saved machine learning artifacts

---

# 🧪 Testing

The project contains automated tests covering important components of the prediction pipeline.

Testing areas include:

* Feature encoding
* Model loading
* Prediction generation
* Risk classification
* Retention recommendation generation
* Bulk prediction validation
* Report functionality

Tests can be executed using:

```bash
python -m unittest discover -s tests -v
```

---

# 📈 Business Value

RetainIQ provides value across three levels of analytics:

### Descriptive Analytics

> What happened?

* Churn trends
* Customer statistics
* Revenue analysis
* Service analysis

### Predictive Analytics

> What is likely to happen?

* Churn prediction
* Churn probability
* Customer risk levels

### Prescriptive Analytics

> What should we do?

* Retention recommendations
* Customer prioritization
* Risk-based actions

This creates the following decision framework:

```text
DATA
 ↓
INSIGHT
 ↓
PREDICTION
 ↓
EXPLANATION
 ↓
ACTION
 ↓
BUSINESS VALUE
```

---

# 🌟 Key Features

* ✅ End-to-end churn prediction pipeline
* ✅ Comprehensive EDA
* ✅ Data preprocessing
* ✅ Feature engineering
* ✅ Multiple machine learning algorithms
* ✅ Model comparison
* ✅ Production Logistic Regression model
* ✅ Churn probability estimation
* ✅ Customer risk segmentation
* ✅ Explainable predictions
* ✅ Retention recommendations
* ✅ Revenue-at-risk analysis
* ✅ Single customer prediction
* ✅ Bulk customer prediction
* ✅ Interactive prediction dashboard
* ✅ Customer search and filtering
* ✅ Analytical reports
* ✅ Power BI dashboard
* ✅ Automated testing
* ✅ Git/GitHub version control
* ✅ Cloud deployment

---

# ⚠️ Limitations

Although RetainIQ provides an end-to-end predictive solution, several limitations should be considered:

1. The model is trained using a historical telecom customer dataset.
2. Model performance depends on the quality and representativeness of the available data.
3. Historical customer behavior may not perfectly represent future customer behavior.
4. Predictions represent statistical probabilities rather than guaranteed outcomes.
5. Retention recommendations are decision-support suggestions and should be validated by business teams.
6. Revenue-at-risk estimates should not be interpreted as guaranteed revenue loss.
7. Continuous real-time customer behavior data is not currently integrated.
8. Automated model retraining and drift monitoring are not yet implemented.

---

# 🚀 Future Enhancements

Potential future improvements include:

### Machine Learning

* Advanced hyperparameter optimization
* Ensemble model optimization
* Deep learning-based churn prediction
* Cost-sensitive learning
* Probability calibration

### Explainable AI

* Enhanced model-specific SHAP analysis
* Interactive individual explanations
* Counterfactual explanations
* What-if customer analysis

### Retention Intelligence

* Advanced recommendation engine
* Personalized retention strategies
* Customer Lifetime Value prediction
* Retention campaign optimization
* A/B testing of retention strategies

### MLOps

* Automated model retraining
* Model monitoring
* Data drift detection
* Model performance monitoring
* Automated ML pipelines

### Integration

* CRM integration
* Customer support system integration
* Real-time prediction API
* Cloud database integration
* Automated notification systems

---

# 📚 Project Contributions

RetainIQ combines multiple disciplines into one integrated platform:

```text
Data Science
     +
Machine Learning
     +
Explainable AI
     +
Business Intelligence
     +
Web Development
     +
Deployment
     =
RetainIQ
```

The major contribution of the project is the transformation of customer churn prediction from a simple classification problem into a **customer retention intelligence workflow**.

---

# 📖 References and Technologies

The project makes use of the following technologies and resources:

* IBM Telco Customer Churn Dataset
* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Joblib
* Flask
* Microsoft Power BI
* Git
* GitHub
* Render

---

# 📄 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

**Logesh**

B.Sc. Data Science Student

**Project:** RetainIQ — Predictive Customer Retention Intelligence Platform

---

## ⭐ Project Summary

> **RetainIQ is an explainable machine learning-based customer retention intelligence platform that predicts customer churn, estimates risk, explains predictions, recommends retention actions, and provides business intelligence dashboards to support proactive customer retention decisions.**
