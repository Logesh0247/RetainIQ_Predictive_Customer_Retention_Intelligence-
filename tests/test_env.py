"""Optional DB env check — does not fail the unit suite."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if __name__ == "__main__":
    print("Server:", os.getenv("DB_SERVER"))
    print("Database:", os.getenv("DB_DATABASE"))
    print("Driver:", os.getenv("DB_DRIVER"))
