"""Focused checks for encoding, model load, scoring, recommendations, and CSV validation."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import build_feature_frame, preprocess_single, PreprocessingError
from utils.prediction import load_model, is_model_available, predict_customer
from utils.recommendation import generate_recommendation, retention_action_label
from utils.bulk_prediction import run_bulk_prediction_from_bytes, BulkPredictionError
import pandas as pd


MALE_SENIOR_PHONE = {
    "customerID": "TEST-001",
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 8,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 700,
}


class EncodingTests(unittest.TestCase):
    def test_gender_senior_phone_flags(self):
        frame, _ = preprocess_single(MALE_SENIOR_PHONE)
        self.assertEqual(int(frame["Gender_Male"].iloc[0]), 1)
        self.assertEqual(int(frame["Senior Citizen_Yes"].iloc[0]), 1)
        self.assertEqual(int(frame["Phone Service_Yes"].iloc[0]), 1)
        self.assertEqual(frame.shape[1], 30)


class ModelTests(unittest.TestCase):
    def test_best_model_is_logistic_regression(self):
        self.assertTrue(is_model_available())
        bundle = load_model()
        self.assertEqual(bundle["name"], "Logistic Regression")
        self.assertTrue(hasattr(bundle["model"], "predict_proba"))


class PredictTests(unittest.TestCase):
    def test_predict_customer_shape(self):
        result = predict_customer(MALE_SENIOR_PHONE)
        self.assertIn("probability", result)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIn(result["risk_level"], {"Low Risk", "Medium Risk", "High Risk"})
        self.assertTrue(result["top_drivers"])


class RecommendationTests(unittest.TestCase):
    def test_month_to_month_gets_contract_offer(self):
        recs = generate_recommendation(
            pd.Series(MALE_SENIOR_PHONE),
            0.72,
            "High Risk",
        )
        self.assertTrue(recs)
        joined = " ".join(recs).lower()
        self.assertIn("contract", joined)

    def test_high_risk_action_label(self):
        label = retention_action_label(pd.Series(MALE_SENIOR_PHONE), 0.8, "High Risk")
        self.assertEqual(label, "Offer 15% Discount")


class BulkTests(unittest.TestCase):
    def test_missing_columns_rejected(self):
        raw = b"customerID,gender\nA,Male\n"
        with self.assertRaises(BulkPredictionError):
            run_bulk_prediction_from_bytes(raw, "bad.csv")


if __name__ == "__main__":
    unittest.main()
