from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

load_dotenv(ROOT_DIR / ".env")

FACEIT_API_KEY = os.getenv("FACEIT_API_KEY", "")
FACEIT_GAME = os.getenv("FACEIT_GAME", "cs2")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'faceit.db'}")
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.35"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
FORM_WINDOW = int(os.getenv("FORM_WINDOW", "20"))
FACEIT_BASE_URL = "https://open.faceit.com/data/v4"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
