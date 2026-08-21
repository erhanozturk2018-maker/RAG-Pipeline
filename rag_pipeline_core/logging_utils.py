"""
Daily-rotating CSV audit log for ingestion and query operations.

One file per calendar day under logs/, named YYYY-MM-DD.csv. The header
row is written when a day's file is first created; later operations on
the same day append to it. "Rotation" is therefore implicit -- the
filename is derived from today's date on every write, so the first
operation after midnight naturally starts a new file. No background
thread, no size checks, no cleanup of old files.

Named logging_utils.py rather than logging.py on purpose: a module named
logging.py inside a package is importable as `rag_pipeline_core.logging`,
which is harmless on its own, but it invites confusion with the stdlib
`logging` module and breaks if this package is ever put directly on
sys.path.

Timestamps are recorded in LOCAL time (the host's timezone), with a UTC
offset included in the ISO 8601 string (e.g. 2026-08-21T10:09:30+03:00).
Local time was chosen over UTC because these logs are meant to be read
next to the dashboard's own operator-facing timeline; the explicit offset
keeps them unambiguous when the container's TZ differs from the host's.

Writing to this log must never take down the pipeline operation it is
recording -- see log_event(), which swallows every exception it can and
degrades to a printed warning.
"""

import csv
from datetime import datetime
from pathlib import Path

from rag_pipeline_core import config

# Column order is part of this module's contract -- downstream readers
# (e.g. the dashboard) index by position as well as by name, so append
# new columns at the END rather than inserting them in the middle.
FIELDNAMES = [
    "timestamp",
    "event_type",
    "user",
    "model",
    "embedding_model",
    "chunk_count",
    "file",
    "query",
    "description",
    "status",
    "error",
    "duration_ms",
]

LOG_DIR = config.BASE_DIR / "logs"

# Used whenever a caller doesn't supply a user. There's no auth system
# yet; the API and CLI both accept an optional `user` so a caller (like
# the dashboard) can pass a real identifier once there is one.
DEFAULT_USER = "local"


def _today_log_path() -> Path:
    """Path to the current day's CSV file (does not create anything)."""
    return LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.csv"


def log_event(
    event_type: str,
    *,
    user: str | None = None,
    model: str = "",
    embedding_model: str = "",
    chunk_count: int | str = "",
    file: str = "",
    query: str = "",
    description: str = "",
    status: str = "success",
    error: str = "",
    duration_ms: int | str = "",
) -> None:
    """Append one row to today's audit log.

    Every argument is keyword-only except event_type, so call sites read
    as self-documenting field assignments and adding a column later
    can't silently shift an existing positional argument.

    This function never raises. A failure to log (unwritable logs/
    directory, disk full, permissions) prints a warning and returns --
    an audit log is not worth crashing a successful ingestion over.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _today_log_path()
        # Check before opening in append mode: opening the file is what
        # creates it, so afterwards it always exists.
        is_new_file = not path.exists()

        # newline="" is required by the csv module -- it does its own
        # line-ending handling, and without this Windows would turn
        # every row terminator into \r\r\n.
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if is_new_file:
                writer.writeheader()
            # DictWriter handles quoting/escaping for commas, quotes and
            # embedded newlines in free-text fields (query, description,
            # error) -- this is why the csv module is used instead of
            # joining strings by hand.
            writer.writerow(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event_type": event_type,
                    "user": user or DEFAULT_USER,
                    "model": model,
                    "embedding_model": embedding_model,
                    "chunk_count": chunk_count,
                    "file": file,
                    "query": query,
                    "description": description,
                    "status": status,
                    "error": error,
                    "duration_ms": duration_ms,
                }
            )
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"[logging_utils] WARNING: failed to write audit log entry: {e}")
