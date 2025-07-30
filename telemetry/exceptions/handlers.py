"""
Exception handling decorators and handlers for the YT Summariser application.

This module provides decorators and context managers for handling exceptions
in a consistent and robust manner across the application.
"""

import asyncio
import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast

from .custom_exceptions import BaseYTSummarizerError, ExternalServiceError, VideoProcessingError
from .context import log_exception

from ..logging import get_logger

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Module logger
logger = get_logger(__name__)


def handle_exceptions(
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    reraise: bool = True,
    default_return: Optional[T] = None,
    include_traceback: bool = True,
    context_message: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to handle exceptions with logging and optional re-raising.
    
    Args:
        exceptions: Exception types to catch
        logger_instance: Logger to use
        level: Logging level for exceptions
        reraise: Whether to re-raise the exception after logging
        default_return: Default value to return if not re-raising
        include_traceback: Whether to include traceback in logs
        context_message: Additional context message for the error
    
    Returns:
        Decorated function
    
    Example:
        >>> @handle_exceptions(VideoProcessingError, reraise=False, default_return={})
        ... def process_video(video_id: str) -> dict:
        ...     # Processing logic
        ...     raise VideoProcessingError("Failed to download", video_id=video_id)
        
        >>> # For async functions
        >>> @handle_exceptions(ExternalServiceError)
        ... async def fetch_data(url: str) -> dict:
        ...     # Async logic
        ...     raise ExternalServiceError("API", "Rate limited", status_code=429)
    """
    def decorator(func: F) -> F:
        # Handle async functions
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    log = logger_instance or get_logger(func.__module__)
                    
                    # Build context
                    context = {
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs
                    }
                    if context_message:
                        context["message"] = context_message
                    
                    # Log the exception
                    log_exception(e, log, level, include_traceback, context)
                    
                    if reraise:
                        raise
                    return default_return
            
            return cast(F, async_wrapper)
        
        # Handle sync functions
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    log = logger_instance or get_logger(func.__module__)
                    
                    # Build context
                    context = {
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs
                    }
                    if context_message:
                        context["message"] = context_message
                    
                    # Log the exception
                    log_exception(e, log, level, include_traceback, context)
                    
                    if reraise:
                        raise
                    return default_return
            
            return cast(F, sync_wrapper)
    
    return decorator




@contextmanager
def handle_api_errors(service_name: str, raise_on_error: bool = True):
    """
    Context manager for handling API errors with consistent logging.
    
    Args:
        service_name: Name of the API service
        raise_on_error: Whether to raise exceptions or suppress them
    
    Example:
        >>> with handle_api_errors("YouTube API"):
        ...     response = youtube_client.get_video_info(video_id)
        ...     # Any API errors will be logged and converted to ExternalServiceError
    """
    try:
        yield
    except Exception as e:
        # Check for common API error patterns
        error_details = {
            "service": service_name,
            "original_error": type(e).__name__
        }
        
        # Handle different error types
        if hasattr(e, 'response'):
            # HTTP errors (requests library)
            response = getattr(e, 'response')
            if hasattr(response, 'status_code'):
                error_details['status_code'] = response.status_code
            if hasattr(response, 'text'):
                error_details['response_body'] = response.text[:500]  # Limit size
        
        elif hasattr(e, 'status_code'):
            # Direct status code attribute
            error_details['status_code'] = e.status_code
        
        # Log the error
        logger.error(f"{service_name} error: {str(e)}", extra=error_details, exc_info=True)
        
        if raise_on_error:
            # Convert to ExternalServiceError
            raise ExternalServiceError(
                service=service_name,
                message=str(e),
                **error_details
            ) from e


def handle_timeout(timeout_seconds: float, error_message: Optional[str] = None):
    """
    Decorator to handle timeouts for long-running operations.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        error_message: Custom error message for timeout
    
    Returns:
        Decorated function
    
    Example:
        >>> @handle_timeout(30.0, "Video processing timed out")
        ... async def process_large_video(video_id: str):
        ...     # Long-running operation
        ...     await asyncio.sleep(60)  # This will timeout
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError as e:
                    msg = error_message or f"{func.__name__} timed out after {timeout_seconds}s"
                    logger.error(msg, extra={
                        "function": func.__name__,
                        "timeout": timeout_seconds,
                        "args": args,
                        "kwargs": kwargs
                    })
                    raise VideoProcessingError(
                        msg,
                        step=func.__name__,
                        timeout_seconds=timeout_seconds
                    ) from e
            
            return cast(F, async_wrapper)
        else:
            raise ValueError("handle_timeout decorator only supports async functions")
    
    return decorator




# Export all public handlers and decorators
__all__ = [
    'handle_exceptions',
    'handle_api_errors',
    'handle_timeout'
]