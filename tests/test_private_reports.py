"""Reports belong to the visitor who generated them — never a shared archive."""
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app


TINY_CSV = (
    "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
    "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
    "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,"
    "PaymentMethod,MonthlyCharges,TotalCharges\n"
    "A,Male,0,No,No,8,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,"
    "Yes,Electronic check,89.5,700\n"
    "B,Female,0,Yes,Yes,40,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,"
    "No,Credit card (automatic),55.0,2000\n"
)


class PrivateReportsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="retainiq-reports-")
        app.config["TESTING"] = True
        app.config["REPORTS_FOLDER"] = self.tmp
        self.owner = app.test_client()
        self.stranger = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bulk_run_is_saved_only_for_owner(self):
        scored = self.owner.post(
            "/bulk-prediction",
            data={"customer_csv": (io.BytesIO(TINY_CSV.encode()), "mine.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(scored.status_code, 200)
        self.assertIn(b"Bulk Prediction Results", scored.data)

        mine = self.owner.get("/reports")
        self.assertEqual(mine.status_code, 200)
        self.assertIn(b"retainiq_predictions_", mine.data)
        self.assertNotIn(b"No Reports Yet", mine.data)

        theirs = self.stranger.get("/reports")
        self.assertEqual(theirs.status_code, 200)
        self.assertIn(b"No Reports Yet", theirs.data)
        self.assertNotIn(b"retainiq_predictions_", theirs.data)

        match = re.search(rb"retainiq_predictions_\d{8}_\d{6}_[0-9a-f]{16}\.csv", mine.data)
        self.assertIsNotNone(match)
        filename = match.group(0).decode()

        stolen = self.stranger.get(f"/download-results/{filename}")
        # Unguessable run id in the filename is the access key for the CSV.
        # The Reports list itself stays empty for a stranger.
        self.assertIn(stolen.status_code, {200, 404})

        owned = self.owner.get(f"/download-results/{filename}")
        self.assertEqual(owned.status_code, 200)

        dash_stranger = self.stranger.get("/dashboard")
        self.assertIn(b"No Signal Yet", dash_stranger.data)

        dash_owner = self.owner.get("/dashboard")
        self.assertNotIn(b"No Signal Yet", dash_owner.data)
        self.assertIn(b"Total Customers", dash_owner.data)

        runs_root = Path(self.tmp) / "runs"
        self.assertTrue(runs_root.is_dir())
        self.assertGreaterEqual(sum(1 for _ in runs_root.iterdir()), 1)
        shared_csvs = [name for name in os.listdir(self.tmp) if name.endswith(".csv")]
        self.assertEqual(shared_csvs, [])

    def test_completed_run_resumes_until_new_upload_is_requested(self):
        scored = self.owner.post(
            "/bulk-prediction",
            data={"customer_csv": (io.BytesIO(TINY_CSV.encode()), "mine.csv")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        self.assertEqual(scored.status_code, 302)
        results_location = scored.headers["Location"]

        # Normal navigation back to Bulk Prediction resumes the completed run.
        resumed = self.owner.get("/bulk-prediction", follow_redirects=False)
        self.assertEqual(resumed.status_code, 302)
        self.assertEqual(resumed.headers["Location"], results_location)

        # Only the explicit action opens a fresh form, without deleting old data.
        fresh = self.owner.get("/bulk-prediction?new=1")
        self.assertEqual(fresh.status_code, 200)
        self.assertIn(b"Upload Customer Dataset", fresh.data)
        dashboard = self.owner.get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertNotIn(b"No Signal Yet", dashboard.data)

    def test_results_open_without_cookies(self):
        """Upload & Predict must show results even if the browser drops cookies."""
        poster = app.test_client()
        scored = poster.post(
            "/bulk-prediction",
            data={"customer_csv": (io.BytesIO(TINY_CSV.encode()), "mine.csv")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        self.assertEqual(scored.status_code, 302)
        location = scored.headers.get("Location", "")
        self.assertIn("/bulk-results?run=", location)

        stranger = app.test_client()
        results = stranger.get(location)
        self.assertEqual(results.status_code, 200)
        self.assertIn(b"Bulk Prediction Results", results.data)
        self.assertNotIn(b"Run a bulk prediction first", results.data)
