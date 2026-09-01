"""End-to-end checks for the rebuilt Single Prediction experience."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as retainiq_app
from utils.examples import get_example
from utils.prediction import (
    predict_customer, explain_prediction, simulate_what_if, calculate_risk_level,
)


class ExampleCustomerTests(unittest.TestCase):
    def test_examples_are_form_ready(self):
        high = get_example("high")
        low = get_example("low")
        for example in (high, low):
            for field in ("gender", "tenure", "Contract", "MonthlyCharges"):
                self.assertIn(field, example)
                self.assertNotEqual(example[field], "")
        self.assertGreater(
            predict_customer(high)["probability"],
            predict_customer(low)["probability"],
        )

    def test_unknown_kind_falls_back(self):
        self.assertIn("Contract", get_example("not-a-kind"))


class ExplanationTests(unittest.TestCase):
    def test_signed_contributions(self):
        result = predict_customer(get_example("high"))
        drivers = explain_prediction(result["feature_frame"])
        self.assertTrue(drivers)
        for driver in drivers:
            self.assertIn(driver["direction"], ("risk", "protective"))
            self.assertLessEqual(driver["share"], 100.0)
        # Ranked by absolute impact.
        shares = [driver["share"] for driver in drivers]
        self.assertEqual(shares, sorted(shares, reverse=True))


class WhatIfTests(unittest.TestCase):
    def test_two_year_contract_reduces_risk(self):
        result = predict_customer(get_example("high"))
        scenarios = simulate_what_if(result["cleaned_record"], result["probability"])
        self.assertTrue(scenarios)
        by_id = {item["id"]: item for item in scenarios}
        self.assertIn("contract_two_year", by_id)
        self.assertTrue(by_id["contract_two_year"]["improves"])
        self.assertLess(by_id["contract_two_year"]["probability"], result["probability"])
        # Sorted best-first.
        deltas = [item["delta_points"] for item in scenarios]
        self.assertEqual(deltas, sorted(deltas))

    def test_risk_bands_are_shared(self):
        from utils import universal_churn
        self.assertIs(universal_churn.calculate_risk_level, calculate_risk_level)


class SinglePredictionRouteTests(unittest.TestCase):
    def setUp(self):
        retainiq_app.app.config["TESTING"] = True
        self.client = retainiq_app.app.test_client()

    def test_example_prefills_the_form(self):
        response = self.client.get("/single-prediction?example=high")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"example-chip", response.data)
        self.assertIn(b"Score one customer", response.data)

    def test_post_redirects_to_a_stable_result_page(self):
        payload = get_example("high")
        response = self.client.post("/single-prediction", data=payload)
        self.assertEqual(response.status_code, 302)
        location = response.headers["Location"]
        self.assertRegex(location, r"/single-prediction/result/[0-9a-f]{24}$")

        # The result page is a GET, so it survives a refresh.
        for _ in range(2):
            page = self.client.get(location)
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"What-if simulator", page.data)
            self.assertIn(b"Why this score", page.data)

        token = location.rsplit("/", 1)[-1]
        download = self.client.get(f"/single-prediction/result/{token}/download")
        self.assertEqual(download.status_code, 200)
        body = download.data.decode("utf-8")
        self.assertIn("Churn Probability (%)", body)
        self.assertIn("Best What-If Action", body)

    def test_invalid_submission_returns_the_form_with_an_error(self):
        response = self.client.post("/single-prediction", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Missing required field", response.data)

    def test_expired_scorecard_redirects_to_the_form(self):
        response = self.client.get("/single-prediction/result/" + "0" * 24)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/single-prediction", response.headers["Location"])


if __name__ == "__main__":
    unittest.main()
