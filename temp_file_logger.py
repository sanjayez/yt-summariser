import os
import threading
from datetime import datetime, timezone

_lock = threading.Lock()


def append_line(file_path: str, message: str) -> None:
    """Append a single log line to the given file (best-effort, thread-safe).

    - Ensures parent directory exists (if provided)
    - Prepends an ISO UTC timestamp
    - Swallows all exceptions to avoid impacting callers
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {message}\n"

        with _lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        # Best-effort logging; never raise
        pass

