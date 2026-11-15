"""Minimal local .env loader for credentials; values are never printed or logged."""

from pathlib import Path
import os
import re

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_env_file(path: Path) -> list[str]:
    """Set KEY=VALUE pairs from ``path`` without overriding existing variables.

    Returns the names that were set. A missing file is not an error. Empty values
    are skipped so a placeholder line still triggers the normal missing-key error.
    """
    if not path.is_file():
        return []
    loaded = []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not _KEY.fullmatch(key):
            raise ValueError(f"{path.name} line {number}: expected KEY=VALUE")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded
