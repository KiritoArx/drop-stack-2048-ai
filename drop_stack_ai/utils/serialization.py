import os
from typing import Any

from flax.serialization import to_bytes, from_bytes


def save_params(params: Any, path: str) -> None:
    """Save model parameters to ``path`` using Flax serialization."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(to_bytes(params))


def load_params(path: str, target: Any) -> Any:
    """Load parameters from ``path`` into ``target`` structure."""
    with open(path, "rb") as f:
        data = f.read()
    return from_bytes(target, data)
