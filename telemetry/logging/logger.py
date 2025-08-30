"""Core logger functionality for the YT Summariser application."""

import logging
import logging.handlers
import sys
from pathlib import Path

from .formatters import JSONFormatter

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: str | int = logging.INFO,
    format_string: str | None = None,
    date_format: str | None = None,
) -> logging.Logger:
    """Get a configured logger instance with consistent formatting."""
    logger = logging.getLogger(name)

    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Set formatter
    formatter = logging.Formatter(
        fmt=format_string or DEFAULT_FORMAT, datefmt=date_format or DEFAULT_DATE_FORMAT
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def setup_logging(
    level: str | int = logging.INFO,
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False,
    format_string: str | None = None,
    date_format: str | None = None,
    disable_existing_loggers: bool = False,
) -> None:
    """Configure global logging settings for the application."""
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt=format_string or DEFAULT_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT,
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure existing loggers
    if not disable_existing_loggers:
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = True


def get_child_logger(parent_logger: logging.Logger, name: str) -> logging.Logger:
    """Create a child logger with the parent's configuration."""
    return parent_logger.getChild(name)


def basic_config(level: str = "INFO", format: str | None = None) -> None:
    """Quick logging setup for scripts and notebooks."""
    setup_logging(level=level, format_string=format)
