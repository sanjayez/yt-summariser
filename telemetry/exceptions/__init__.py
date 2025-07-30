"""
Exception handling utilities for the YT Summariser application.

This package provides:
- Custom exception classes for different error scenarios
- Decorators for exception handling and context addition
- Utility functions for logging exceptions
- Common exception handlers for API errors, timeouts, etc.

The package is organized into focused modules:
- custom_exceptions: All custom exception classes
- handlers: Exception handling decorators and context managers
- context: Error context, logging, and utility functions
"""

# Import all exception classes
from .custom_exceptions import (
    BaseYTSummarizerError,
    VideoProcessingError,
    ExternalServiceError,
    ValidationError
)

# Import all handlers and decorators
from .handlers import (
    handle_exceptions,
    handle_api_errors,
    handle_timeout
)

# Import retry functionality
from .retry import (
    retry_on_exception
)

# Import all context and utility functions
from .context import (
    log_exception,
    format_exception_chain,
    get_error_summary,
    safe_cleanup,
    with_error_context
)

# Export all public APIs - maintains backward compatibility
__all__ = [
    # Exception classes
    'BaseYTSummarizerError',
    'VideoProcessingError',
    'ExternalServiceError',
    'ValidationError',
    
    # Decorators
    'handle_exceptions',
    'with_error_context',
    'handle_timeout',
    'retry_on_exception',
    
    # Context managers
    'handle_api_errors',
    
    # Functions
    'log_exception',
    'format_exception_chain',
    'get_error_summary',
    'safe_cleanup'
]