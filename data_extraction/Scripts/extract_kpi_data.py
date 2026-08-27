import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from db_connection import get_engine
from src.paths import EXTRACTION_DIR

engine = get_engine()
query = (EXTRACTION_DIR / "queries" / "kpi_data.sql").read_text(encoding="utf-8")
df = pd.read_sql(query, engine)
print(df)
df.to_csv(EXTRACTION_DIR / "raw_data" / "kpi_data.csv", index=False)
print("KPI Data Extracted Successfully")
