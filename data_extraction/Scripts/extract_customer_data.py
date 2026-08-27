import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from db_connection import get_engine
from src.paths import EXTRACTION_DIR

engine = get_engine()

query = (EXTRACTION_DIR / "queries" / "customer_data.sql").read_text(encoding="utf-8")
df = pd.read_sql(query, engine)

print(df.head())
print("\nRows:", len(df))
print("Columns:", len(df.columns))

out = EXTRACTION_DIR / "raw_data" / "customer_data.csv"
df.to_csv(out, index=False)
print("\nCustomer data extracted successfully")
