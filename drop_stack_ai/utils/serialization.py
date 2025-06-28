import os
from typing import Any

from google.cloud import storage

from flax.serialization import to_bytes, from_bytes


def _write_bytes(path: str, data: bytes) -> None:
    if path.startswith("gs://"):
        bucket, blob_name = path[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket)
        bucket.blob(blob_name).upload_from_string(data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def save_params(params: Any, path: str) -> None:
    """Save model parameters to ``path`` using Flax serialization."""
    _write_bytes(path, to_bytes(params))


def _read_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        bucket, blob_name = path[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket)
        return bucket.blob(blob_name).download_as_bytes()
    else:
        with open(path, "rb") as f:
            return f.read()


def load_params(path: str, target: Any) -> Any:
    """Load parameters from ``path`` into ``target`` structure."""
    data = _read_bytes(path)
    return from_bytes(target, data)


def load_bytes(path: str) -> bytes:
    """Load raw bytes from ``path`` which may be local or ``gs://``."""
    return _read_bytes(path)


def save_bytes(data: bytes, path: str) -> None:
    """Write raw bytes to ``path`` which may be local or ``gs://``."""
    _write_bytes(path, data)


def upload_file(local_path: str, gcs_path: str) -> None:
    """Upload ``local_path`` to ``gcs_path`` using Google Cloud Storage."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    if not gcs_path.startswith("gs://"):
        raise ValueError("Destination must be a gs:// path")
    bucket, blob_name = gcs_path[5:].split("/", 1)
    client = storage.Client()
    client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)


def download_file(gcs_path: str, local_path: str) -> None:
    """Download ``gcs_path`` to ``local_path`` using Google Cloud Storage."""
    if not gcs_path.startswith("gs://"):
        raise ValueError("Source must be a gs:// path")
    bucket, blob_name = gcs_path[5:].split("/", 1)
    client = storage.Client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.bucket(bucket).blob(blob_name).download_to_filename(local_path)
