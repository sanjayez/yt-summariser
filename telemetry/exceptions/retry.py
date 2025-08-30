"""
Retry functionality for exception handling.

This module provides retry decorators and utilities for handling transient failures
and implementing robust retry logic with exponential backoff.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from ..logging import get_logger

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])


def retry_on_exception(
    max_attempts: int = 3,
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    logger_instance: logging.Logger | None = None,
) -> Callable[[F], F]:
    """
    Decorator to retry function execution on specific exceptions.

    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Exception types to retry on
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        logger_instance: Logger to use for retry messages

    Returns:
        Decorated function

    Example:
        >>> @retry_on_exception(max_attempts=3, exceptions=ExternalServiceError, delay=2.0)
        ... async def fetch_api_data(endpoint: str) -> dict:
        ...     # This will retry up to 3 times if ExternalServiceError is raised
        ...     response = await api_client.get(endpoint)
        ...     return response.json()
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                log = logger_instance or get_logger(func.__module__)
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts - 1:
                            # Last attempt, re-raise
                            log.error(
                                f"{func.__name__} failed after {max_attempts} attempts",
                                extra={"attempts": max_attempts, "final_error": str(e)},
                            )
                            raise

                        # Log retry attempt
                        log.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {current_delay}s: {str(e)}",
                            extra={
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "delay": current_delay,
                                "error": str(e),
                            },
                        )

                        # Wait before retry
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor

                # Should never reach here
                return None

            return cast(F, async_wrapper)
        else:
            # Sync version
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                log = logger_instance or get_logger(func.__module__)
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts - 1:
                            # Last attempt, re-raise
                            log.error(
                                f"{func.__name__} failed after {max_attempts} attempts",
                                extra={"attempts": max_attempts, "final_error": str(e)},
                            )
                            raise

                        # Log retry attempt
                        log.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {current_delay}s: {str(e)}",
                            extra={
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "delay": current_delay,
                                "error": str(e),
                            },
                        )

                        # Wait before retry
                        time.sleep(current_delay)
                        current_delay *= backoff_factor

                # Should never reach here
                return None

            return cast(F, sync_wrapper)

    return decorator


# Export retry functionality
__all__ = ["retry_on_exception"]
