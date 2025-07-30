"""
Logging utilities for the YT Summariser application.

This module provides production-ready logging configuration with support for:
- Consistent formatting across the application
- Multiple log levels and handlers
- File logging with rotation
- JSON logging for production environments
- Function call debugging decorators
"""

import functools
import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])

# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if any
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'pathname', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'getMessage']:
                log_data[key] = value
        
        return json.dumps(log_data)


def get_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Logging level (default: INFO)
        format_string: Custom format string (default: DEFAULT_FORMAT)
        date_format: Custom date format (default: DEFAULT_DATE_FORMAT)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
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
        fmt=format_string or DEFAULT_FORMAT,
        datefmt=date_format or DEFAULT_DATE_FORMAT
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    disable_existing_loggers: bool = False
) -> None:
    """
    Configure global logging settings for the application.
    
    Args:
        level: Global logging level
        log_file: Path to log file (enables file logging if provided)
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        json_format: Use JSON formatting for production (default: False)
        format_string: Custom format string (ignored if json_format=True)
        date_format: Custom date format
        disable_existing_loggers: Whether to disable existing loggers
    
    Example:
        >>> # Development setup
        >>> setup_logging(level="DEBUG")
        
        >>> # Production setup with JSON logging
        >>> setup_logging(
        ...     level="INFO",
        ...     log_file="/var/log/app/yt-summariser.log",
        ...     json_format=True
        ... )
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt=format_string or DEFAULT_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT
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
            encoding='utf-8'
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


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    include_result: bool = True,
    include_args: bool = True,
    include_kwargs: bool = True,
    max_length: int = 200
) -> Callable[[F], F]:
    """
    Decorator to log function calls for debugging.
    
    Args:
        logger: Logger instance to use (uses function's module logger if None)
        level: Logging level for the messages
        include_result: Whether to log the function's return value
        include_args: Whether to log positional arguments
        include_kwargs: Whether to log keyword arguments
        max_length: Maximum length of logged values (truncates if longer)
    
    Returns:
        Decorated function
    
    Example:
        >>> @log_function_call(level=logging.INFO)
        ... def process_video(video_id: str, quality: str = "720p") -> dict:
        ...     return {"status": "processed", "id": video_id}
        
        >>> # This will log:
        >>> # INFO - Calling process_video(args=('abc123',), kwargs={'quality': '1080p'})
        >>> # INFO - process_video returned: {'status': 'processed', 'id': 'abc123'}
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get logger for the function's module if not provided
            func_logger = logger or logging.getLogger(func.__module__)
            
            # Prepare log message for function call
            log_parts = [f"Calling {func.__name__}("]
            
            if include_args and args:
                args_str = str(args)
                if len(args_str) > max_length:
                    args_str = args_str[:max_length] + "..."
                log_parts.append(f"args={args_str}")
            
            if include_kwargs and kwargs:
                if include_args and args:
                    log_parts.append(", ")
                kwargs_str = str(kwargs)
                if len(kwargs_str) > max_length:
                    kwargs_str = kwargs_str[:max_length] + "..."
                log_parts.append(f"kwargs={kwargs_str}")
            
            log_parts.append(")")
            func_logger.log(level, "".join(log_parts))
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log the result
                if include_result:
                    result_str = str(result)
                    if len(result_str) > max_length:
                        result_str = result_str[:max_length] + "..."
                    func_logger.log(
                        level,
                        f"{func.__name__} returned: {result_str}"
                    )
                
                return result
                
            except Exception as e:
                # Log the exception
                func_logger.exception(
                    f"{func.__name__} raised exception: {type(e).__name__}: {str(e)}"
                )
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def get_child_logger(parent_logger: logging.Logger, name: str) -> logging.Logger:
    """
    Create a child logger with the parent's configuration.
    
    Args:
        parent_logger: Parent logger instance
        name: Child logger name suffix
    
    Returns:
        Child logger instance
    
    Example:
        >>> parent = get_logger("myapp")
        >>> child = get_child_logger(parent, "submodule")
        >>> # Child logger name will be "myapp.submodule"
    """
    return parent_logger.getChild(name)


def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    message_template: str = "{func_name} executed in {elapsed:.3f} seconds"
) -> Callable[[F], F]:
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use
        level: Logging level for the message
        message_template: Template for the log message
    
    Returns:
        Decorated function
    """
    import time
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = logger or logging.getLogger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                func_logger.log(
                    level,
                    message_template.format(
                        func_name=func.__name__,
                        elapsed=elapsed
                    )
                )
        
        return wrapper  # type: ignore
    
    return decorator


# Convenience function for quick setup in scripts
def basic_config(level: str = "INFO", format: Optional[str] = None) -> None:
    """
    Quick logging setup for scripts and notebooks.
    
    Args:
        level: Logging level as string
        format: Optional format string
    
    Example:
        >>> from core.utils.logging_utils import basic_config
        >>> basic_config("DEBUG")
    """
    setup_logging(level=level, format_string=format)