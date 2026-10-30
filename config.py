"""Central configuration and secret loading.

Secrets are read from environment variables (optionally populated from a local
``.env`` file) or from Streamlit secrets. Secret *values* are never returned by
the status helpers and must never be logged, exported or committed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "Fashion & Jewelry Customer Voice Intelligence"
APP_VERSION = "0.1.0"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Where the agent saves its CSV files (git-ignored). Override with FJVI_RUNS_DIR.
RUNS_DIR = Path(os.environ.get("FJVI_RUNS_DIR") or DATA_DIR / "runs")

# The target counts *valid, unique comments after preprocessing* (see preprocessing.py).
DEFAULT_TARGET = 2000
MIN_PRODUCT_MENTIONS = 5
SENTIMENT_MODEL_VERSION = "vader-3.3.2+domain-lexicon-v1"

NICHES = ("Jewelry", "Clothing", "Both")
LANGUAGE_FILTERS = {
    "any": "All languages (no filter)",
    "en": "English only (short / undetermined text is kept)",
}

# Names of credentials the app understands. Values are never stored here.
SECRET_NAMES = {
    "youtube": "YOUTUBE_API_KEY",
    "gcs_bucket": "GCS_BUCKET_NAME",
    "gcs_service_account": "GCS_SERVICE_ACCOUNT_JSON",
    "meta": "META_ACCESS_TOKEN",
    "tiktok": "TIKTOK_RESEARCH_ACCESS_TOKEN",
}


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Existing environment variables always win."""
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv(BASE_DIR / ".env")


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret from the environment, then Streamlit secrets.

    Callers must treat the return value as sensitive: never log or export it.
    """
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit is optional and secrets may not be configured.
        import streamlit as st

        secret = st.secrets.get(name)
        if secret:
            if hasattr(secret, "items"):  # TOML table -> JSON string
                return json.dumps({k: v for k, v in secret.items()})
            return str(secret)
    except Exception:  # noqa: BLE001 - any failure simply means "not configured"
        pass
    return default


def is_configured(name: str) -> bool:
    """True if a credential exists. Never exposes the value."""
    return bool(get_secret(name))
