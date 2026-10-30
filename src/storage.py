"""Optional Google Cloud Storage persistence (separate from YouTube API access).

Never required by the app. Needs: GCS_BUCKET_NAME, GCS_SERVICE_ACCOUNT_JSON (the
JSON content, kept in env / Streamlit secrets) and ``pip install google-cloud-storage``.
Use a private bucket and a least-privilege service account (e.g. Storage Object
Creator + Viewer on that one bucket). A bucket may incur charges. Objects are
written only on an explicit user action; this module never makes anything public.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import SECRET_NAMES, get_secret
from src.schemas import ConnectorError, ConnectorNotConfigured
from src.security import safe_object_name

PREFIX = "exports/"
SETUP = ("Cloud Storage is optional. Set GCS_BUCKET_NAME and GCS_SERVICE_ACCOUNT_JSON (service-account "
         "key JSON) in .env or Streamlit secrets, install google-cloud-storage, and keep the bucket "
         "private. Buckets may incur charges. Otherwise just download the CSVs and re-upload later.")


def is_configured() -> bool:
    return bool(get_secret(SECRET_NAMES["gcs_bucket"]) and get_secret(SECRET_NAMES["gcs_service_account"]))


def _client_and_bucket(client=None):
    bucket_name = get_secret(SECRET_NAMES["gcs_bucket"])
    if not bucket_name:
        raise ConnectorNotConfigured(SETUP)
    if client is not None:
        return client, bucket_name
    info_raw = get_secret(SECRET_NAMES["gcs_service_account"])
    if not info_raw:
        raise ConnectorNotConfigured(SETUP)
    try:
        from google.cloud import storage  # type: ignore
        from google.oauth2 import service_account  # type: ignore
    except ImportError as exc:
        raise ConnectorNotConfigured("Install google-cloud-storage to use Cloud Storage. " + SETUP) from exc
    try:
        creds = service_account.Credentials.from_service_account_info(json.loads(info_raw))
        return storage.Client(credentials=creds, project=creds.project_id), bucket_name
    except Exception as exc:  # noqa: BLE001 - never echo credential content
        raise ConnectorError("The service-account JSON could not be used. Check GCS_SERVICE_ACCOUNT_JSON.") from exc


def save_exports(files: dict[str, bytes], label: str, client=None) -> list[str]:
    """Upload files under ``exports/<timestamp>_<label>/``. Returns the object names written."""
    client, bucket_name = _client_and_bucket(client)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = f"{PREFIX}{stamp}_{safe_object_name(label, extension='')[:-1] or 'export'}/"
    written = []
    try:
        bucket = client.bucket(bucket_name)
        for name, data in files.items():
            ext = name.rsplit(".", 1)[-1]
            obj = folder + safe_object_name(name.rsplit(".", 1)[0], extension=ext)
            ctype = "application/json" if ext == "json" else "text/csv; charset=utf-8"
            bucket.blob(obj).upload_from_string(data, content_type=ctype)
            written.append(obj)
    except ConnectorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ConnectorError(f"Cloud Storage save failed ({type(exc).__name__}). Check the bucket "
                             "name and the service account's permissions.") from exc
    return written


def list_exports(client=None, limit: int = 50) -> list[str]:
    client, bucket_name = _client_and_bucket(client)
    try:
        return [b.name for b in client.list_blobs(bucket_name, prefix=PREFIX, max_results=limit)]
    except Exception as exc:  # noqa: BLE001
        raise ConnectorError(f"Could not list exports ({type(exc).__name__}).") from exc


def load_export(object_name: str, client=None) -> bytes:
    if not object_name.startswith(PREFIX) or ".." in object_name:
        raise ConnectorError("Only files under the exports/ folder can be loaded.")
    client, bucket_name = _client_and_bucket(client)
    try:
        return client.bucket(bucket_name).blob(object_name).download_as_bytes()
    except Exception as exc:  # noqa: BLE001
        raise ConnectorError(f"Could not load that file ({type(exc).__name__}).") from exc
