"""Environment configuration helpers for local .env support."""

from __future__ import annotations

import os

_ENV_LOADED = False


def load_dotenv(path: str = ".env") -> None:
    """Load key=value pairs from .env into os.environ if unset."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if not os.path.isfile(path):
        _ENV_LOADED = True
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _ENV_LOADED = True

