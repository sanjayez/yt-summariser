"""
Logging utilities for the YT Summariser application.

This package provides production-ready logging configuration with support for:
- Consistent formatting across the application
- Multiple log levels and handlers
- File logging with rotation
- JSON logging for production environments
- Function call debugging decorators

Public API (maintains backward compatibility with logging_utils.py):
- get_logger: Get a configured logger instance
- setup_logging: Configure global logging settings
- get_child_logger: Create child loggers
- basic_config: Quick logging setup for scripts
- log_function_call: Decorator to log function calls
- log_execution_time: Decorator to log execution time
- JSONFormatter: Custom JSON formatter for structured logging
- DEFAULT_FORMAT: Default logging format string
- DEFAULT_DATE_FORMAT: Default date format string
"""

# Import all public functions and classes to maintain the same API
from .formatters import JSONFormatter
from .function_logging import log_execution_time, log_function_call
from .logger import (
    DEFAULT_DATE_FORMAT,
    DEFAULT_FORMAT,
    basic_config,
    get_child_logger,
    get_logger,
    setup_logging,
)

__all__ = [
    # Core logger functions
    'get_logger',
    'setup_logging', 
    'get_child_logger',
    'basic_config',
    
    # Function logging decorators
    'log_function_call',
    'log_execution_time',
    
    # Formatters
    'JSONFormatter',
    
    # Constants
    'DEFAULT_FORMAT',
    'DEFAULT_DATE_FORMAT',
]