"""Checks for the Retention Planner campaign simulator."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as retainiq_app
from utils.bulk_prediction import run_sample_prediction
from utils.retention_planner import (
    simulate_campaign, segment_mask, RetentionPlannerError, LEVERS_BY_ID,
)


class CampaignSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = run_sample_prediction()

    def test_segment_filters_narrow_the_cohort(self):
        everyone = segment_mask(self.bundle).sum()
        high = segment_mask(self.bundle, risk="High Risk").sum()
        high_mtm = segment_mask(self.bundle, risk="High Risk", contract="Month-to-month").sum()
        self.assertGreater(everyone, high)
        self.assertGreaterEqual(high, high_mtm)

    def test_campaign_reduces_risk_and_reports_economics(self):
        plan = simulate_campaign(self.bundle, ["contract_1y", "auto_pay"], risk="High Risk")
        self.assertFalse(plan["empty"])
        self.assertGreater(plan["segment_size"], 0)
        self.assertLess(plan["avg_probability_after"], plan["avg_probability_before"])
        self.assertGreater(plan["customers_saved"], 0)
        self.assertGreater(plan["revenue_protected_monthly"], 0)
        self.assertAlmostEqual(
            plan["net_monthly"],
            plan["revenue_protected_monthly"] - plan["campaign_cost_monthly"],
            places=2,
        )
        # Annual figures come from the unrounded monthly value.
        self.assertAlmostEqual(
            plan["revenue_protected_annual"],
            plan["revenue_protected_monthly"] * 12,
            delta=0.5,
        )
        # Every treated customer stays in the detail export.
        self.assertEqual(len(plan["detail_df"]), plan["segment_size"])

    def test_untreated_customers_do_not_drift(self):
        # Only two-year customers: the "upgrade to two year" lever cannot apply.
        plan = simulate_campaign(self.bundle, ["contract_2y"], contract="Two year")
        self.assertEqual(plan["treated_count"], 0)
        self.assertEqual(plan["customers_saved"], 0.0)
        self.assertEqual(plan["campaign_cost_monthly"], 0.0)

    def test_lever_breakdown_covers_each_selected_lever(self):
        levers = ["contract_2y", "tech_support", "loyalty_discount"]
        plan = simulate_campaign(self.bundle, levers, risk="High Risk")
        self.assertEqual({row["id"] for row in plan["lever_breakdown"]}, set(levers))
        nets = [row["net"] for row in plan["lever_breakdown"]]
        self.assertEqual(nets, sorted(nets, reverse=True))
        for row in plan["lever_breakdown"]:
            self.assertEqual(row["label"], LEVERS_BY_ID[row["id"]]["label"])

    def test_no_levers_selected_returns_empty_plan(self):
        plan = simulate_campaign(self.bundle, [], risk="High Risk")
        self.assertTrue(plan["empty"])

    def test_universal_runs_are_rejected_with_a_clear_message(self):
        fake = dict(self.bundle)
        fake["is_universal"] = True
        with self.assertRaises(RetentionPlannerError):
            simulate_campaign(fake, ["contract_1y"])


class PlannerRouteTests(unittest.TestCase):
    def setUp(self):
        retainiq_app.app.config["TESTING"] = True
        self.client = retainiq_app.app.test_client()

    def test_empty_state_without_a_run(self):
        page = self.client.get("/retention-planner")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Retention Planner", page.data)

    def test_old_single_prediction_urls_redirect(self):
        for url in ("/single-prediction", "/single-prediction/result/abc"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302)
            self.assertIn("/retention-planner", response.headers["Location"])

    def test_full_flow_scores_a_portfolio_then_plans(self):
        seeded = self.client.post("/bulk-prediction/sample")
        self.assertEqual(seeded.status_code, 302)

        page = self.client.get(
            "/retention-planner?risk=High+Risk&lever=contract_1y&lever=auto_pay"
        )
        self.assertEqual(page.status_code, 200)
        for fragment in (b"Customers in scope", b"Lever attribution",
                         b"Risk mix shift", b"Revenue protected"):
            self.assertIn(fragment, page.data)

        export = self.client.get(
            "/retention-planner/download?risk=High+Risk&lever=contract_1y&lever=auto_pay"
        )
        self.assertEqual(export.status_code, 200)
        body = export.data.decode("utf-8")
        self.assertIn("Revenue Protected / mo", body)
        self.assertIn("Campaign Levers", body)


if __name__ == "__main__":
    unittest.main()
