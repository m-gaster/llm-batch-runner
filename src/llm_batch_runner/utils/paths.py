import os
from typing import Optional
from warnings import warn

from sqlalchemy.engine import make_url


def ensure_sqlite_dir(db_url: str) -> None:
    """Ensure the parent directory for a file-based SQLite URL exists.

    No-op for in-memory or non-file URLs. Safe to call multiple times.
    """
    try:
        url = make_url(db_url)
        db_path = url.database
        if not db_path or db_path == ":memory:":
            return
        # Only attempt to create a directory if there's a parent component
        dir_path = os.path.dirname(db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except Exception:
        # Best-effort only; callers will see connection errors if this fails
        pass


def teardown_sqlite_file(db_url: str) -> None:
    """Best-effort removal of a file-based SQLite DB for the given URL.

    No-op for in-memory URLs. Warns (does not raise) on failure.
    """
    try:
        url = make_url(db_url)
        db_path: Optional[str] = url.database
        if (
            isinstance(db_path, str)
            and db_path
            and db_path != ":memory:"
            and os.path.exists(db_path)
        ):
            os.remove(db_path)
            print(f"Tore down SQLite DB file at: {db_path}")
    except Exception:
        # Swallow teardown issues silently to avoid masking run success
        warn("SQLite DB teardown failed.")


__all__ = [
    "ensure_sqlite_dir",
    "teardown_sqlite_file",
]
