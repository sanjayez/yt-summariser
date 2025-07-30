"""
Error context, logging, and utility functions for exception handling.
This module provides utilities for logging exceptions, formatting error information,
and managing error context throughout the application.
"""

import asyncio
import functools
import logging
import sys
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from .custom_exceptions import BaseYTSummarizerError

from ..logging import get_logger

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])

# Module logger
logger = get_logger(__name__)


def log_exception(
    exc: BaseException,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    include_traceback: bool = True,
    extra_context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an exception with full context."""
    log = logger_instance or logger
    
    # Build log message
    exc_info = {
        "type": type(exc).__name__,
        "message": str(exc),
        "module": type(exc).__module__
    }
    
    # Add custom exception details if available
    if isinstance(exc, BaseYTSummarizerError):
        exc_info["details"] = exc.details
    
    # Add extra context
    if extra_context:
        exc_info["context"] = extra_context
    
    # Log with or without traceback
    if include_traceback:
        log.log(level, f"Exception occurred: {exc_info}", exc_info=True)
    else:
        log.log(level, f"Exception occurred: {exc_info}")


def format_exception_chain(exc: BaseException, max_depth: int = 10) -> str:
    """Format an exception chain showing all causes."""
    parts = []
    current_exc = exc
    depth = 0
    
    while current_exc and depth < max_depth:
        exc_type = type(current_exc).__name__
        exc_msg = str(current_exc)
        
        if isinstance(current_exc, BaseYTSummarizerError) and current_exc.details:
            details_str = ", ".join(f"{k}={v}" for k, v in current_exc.details.items())
            parts.append(f"{exc_type}: {exc_msg} ({details_str})")
        else:
            parts.append(f"{exc_type}: {exc_msg}")
        
        # Move to cause
        current_exc = getattr(current_exc, '__cause__', None) or \
                     getattr(current_exc, '__context__', None)
        depth += 1
    
    return " -> ".join(parts)


def get_error_summary(exc: BaseException) -> Dict[str, Any]:
    """Get a summary of an exception suitable for logging or API responses."""
    summary = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_module": type(exc).__module__
    }
    
    # Add custom exception details
    if isinstance(exc, BaseYTSummarizerError):
        summary["details"] = exc.details
    
    # Add traceback info (last frame only for brevity)
    tb = traceback.extract_tb(sys.exc_info()[2])
    if tb:
        last_frame = tb[-1]
        summary["location"] = {
            "file": last_frame.filename,
            "line": last_frame.lineno,
            "function": last_frame.name
        }
    
    return summary


async def safe_cleanup(cleanup_func: Callable[[], Any], context: str = "cleanup") -> None:
    """Safely execute cleanup code, logging any errors without raising."""
    try:
        if asyncio.iscoroutinefunction(cleanup_func):
            await cleanup_func()
        else:
            cleanup_func()
    except Exception as e:
        logger.error(f"Error during {context}: {str(e)}", exc_info=True)


def with_error_context(
    context_name: str,
    context_value_param: Optional[Union[str, int]] = None,
    add_to_message: bool = True
) -> Callable[[F], F]:
    """Decorator to add context to errors raised within a function."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract context value
            context_value = None
            if context_value_param is not None:
                if isinstance(context_value_param, int) and len(args) > context_value_param:
                    context_value = args[context_value_param]
                elif isinstance(context_value_param, str) and context_value_param in kwargs:
                    context_value = kwargs[context_value_param]
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add context to custom exceptions
                if isinstance(e, BaseYTSummarizerError):
                    e.details[context_name] = context_value or "unknown"
                
                # Modify message if requested
                if add_to_message and context_value:
                    error_msg = str(e)
                    new_msg = f"{error_msg} [{context_name}: {context_value}]"
                    
                    # For custom exceptions, don't modify the message - just add context to details
                    # The __str__ method will automatically include the details
                    if not isinstance(e, BaseYTSummarizerError):
                        raise type(e)(new_msg) from e
                
                raise
        
        return cast(F, wrapper)
    
    return decorator


# Export all public functions
__all__ = [
    'log_exception',
    'format_exception_chain',
    'get_error_summary',
    'safe_cleanup',
    'with_error_context'
]