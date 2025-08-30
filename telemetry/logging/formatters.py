"""
Custom logging formatters for the YT Summariser application.

This module provides specialized formatters for different logging needs:
- JSONFormatter for structured production logging
- Future custom formatters can be added here
"""

import json
import logging
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if any
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "pathname",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "getMessage",
            ]:
                log_data[key] = value

        return json.dumps(log_data)
