"""Compatibility layer that re-exports helpers from split modules.

This preserves external imports like `from llm_batch_runner.utils import ...`.
"""

from typing import Any

from .db import (
    CREATE_SQL,
    DB_URL_DEFAULT,
    count_status,
    fetch_results,
    init_db,
    key_for,
    seed,
    select_to_run,
    set_done,
    set_failed,
    set_inflight,
)
from .paths import ensure_sqlite_dir, teardown_sqlite_file
from .results import (
    derive_results_db_url,
    export_jsonl,
    shape_return,
    write_results_to_db,
)
from .workers import make_pydantic_ai_worker, resolve_worker

__all__ = [
    # db
    "DB_URL_DEFAULT",
    "CREATE_SQL",
    "key_for",
    "init_db",
    "seed",
    "select_to_run",
    "set_inflight",
    "set_done",
    "set_failed",
    "count_status",
    "fetch_results",
    # results
    "export_jsonl",
    "derive_results_db_url",
    "write_results_to_db",
    "shape_return",
    # workers
    "make_pydantic_ai_worker",
    "resolve_worker",
    # paths
    "ensure_sqlite_dir",
    "teardown_sqlite_file",
    # misc for type usage parity
    "Any",
]
