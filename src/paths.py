"""Repo-root paths so scripts and notebooks do not depend on a Windows username."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
POWER_BI_DIR = REPO_ROOT / "Power_BI_dashboard"
RISK_DIR = REPO_ROOT / "Risk_segmentation"
RETENTION_DIR = REPO_ROOT / "retention_Intelligence"
EXTRACTION_DIR = REPO_ROOT / "data_extraction"
SAMPLE_CUSTOMERS = DATA_DIR / "sample_customers.csv"
