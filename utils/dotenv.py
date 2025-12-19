from __future__ import annotations

import os
from pathlib import Path

_DOTENV_LOADED = False


def _repo_root() -> Path:
    # mats/utils/dotenv.py -> mats/utils -> mats -> repo root
    return Path(__file__).resolve().parents[2]


def load_mats_env() -> list[Path]:
    """
    Load local `.env` files into `os.environ` (if `python-dotenv` is installed).

    Precedence (highest wins):
    1) existing environment variables (shell/IDE)
    2) `MATS_DOTENV_PATH` (if set)
    3) `mats/.env`
    4) `.env`
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return []
    _DOTENV_LOADED = True

    try:
        from dotenv import dotenv_values
    except Exception:
        return []

    root = _repo_root()
    candidates: list[Path] = [
        root / ".env",
        root / "mats" / ".env",
    ]

    explicit = os.environ.get("MATS_DOTENV_PATH", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    loaded: list[Path] = []
    merged: dict[str, str] = {}

    for path in candidates:
        if not path.is_file():
            continue
        values = dotenv_values(path)
        for key, value in values.items():
            if not key or value is None:
                continue
            merged[key] = value
        loaded.append(path)

    # Apply merged values, but never overwrite non-empty env vars.
    for key, value in merged.items():
        current = os.environ.get(key)
        if current is not None and current.strip():
            continue
        os.environ[key] = value

    return loaded

