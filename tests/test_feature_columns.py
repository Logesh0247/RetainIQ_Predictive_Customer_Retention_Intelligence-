import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

feature_columns = joblib.load(ROOT / "models" / "feature_columns.pkl")

if __name__ == "__main__":
    print("Number of Features:", len(feature_columns))
    for i, feature in enumerate(feature_columns, 1):
        print(f"{i}. {feature}")
