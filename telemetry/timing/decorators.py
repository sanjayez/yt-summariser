"""
Timing decorators for performance measurement.

This module provides decorators for timing function execution with support
for both synchronous and asynchronous functions.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, TypeVar, cast
import logging

from ..logging import get_logger

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Default logger for timing operations
logger = get_logger(__name__)


def timed_operation(
    name: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
    threshold_ms: Optional[float] = None
) -> Callable[[F], F]:
    """
    Decorator that times function execution and logs elapsed time.
    
    Works with both synchronous and asynchronous functions. Logs execution
    time in milliseconds for better precision in performance analysis.
    
    Args:
        name: Custom name for the operation (defaults to function name)
        logger_instance: Logger to use (defaults to module logger)
        level: Logging level for timing messages
        include_args: Whether to include function arguments in log
        include_result: Whether to include function result in log
        threshold_ms: Only log if execution time exceeds this threshold (ms)
    
    Returns:
        Decorated function that logs its execution time
    
    Example:
        >>> @timed_operation(name="API call", threshold_ms=1000)
        ... def fetch_data(url: str) -> dict:
        ...     # Simulated API call
        ...     time.sleep(2)
        ...     return {"status": "success"}
        
        >>> # Logs: "API call completed in 2001.23 ms" (only if > 1000ms)
        
        >>> @timed_operation(include_args=True)
        ... async def process_batch(items: list) -> int:
        ...     await asyncio.sleep(0.5)
        ...     return len(items)
        
        >>> # Logs: "process_batch(items=[...]) completed in 501.45 ms"
    """
    def decorator(func: F) -> F:
        operation_name = name or func.__name__
        func_logger = logger_instance or logger
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                
                try:
                    result = await func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    
                    if threshold_ms is None or elapsed_ms >= threshold_ms:
                        _log_timing(
                            func_logger, level, operation_name, elapsed_ms,
                            args if include_args else None,
                            kwargs if include_args else None,
                            result if include_result else None
                        )
                    
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    func_logger.error(
                        f"{operation_name} failed after {elapsed_ms:.2f} ms: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    raise
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    
                    if threshold_ms is None or elapsed_ms >= threshold_ms:
                        _log_timing(
                            func_logger, level, operation_name, elapsed_ms,
                            args if include_args else None,
                            kwargs if include_args else None,
                            result if include_result else None
                        )
                    
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    func_logger.error(
                        f"{operation_name} failed after {elapsed_ms:.2f} ms: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def measure_time(
    func: Callable[[], Any],
    name: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.INFO
) -> tuple[Any, float]:
    """
    Simple function to measure execution time of a callable.
    
    Executes the provided function and returns both its result and the
    elapsed time in milliseconds. Useful for quick timing measurements
    without decorators or context managers.
    
    Args:
        func: Callable to execute and time
        name: Optional name for logging (defaults to function name)
        logger_instance: Logger to use for output
        level: Logging level for timing message
    
    Returns:
        Tuple of (function result, elapsed time in milliseconds)
    
    Example:
        >>> def heavy_computation():
        ...     return sum(i**2 for i in range(1000000))
        
        >>> result, elapsed_ms = measure_time(heavy_computation)
        >>> print(f"Computation result: {result}, Time: {elapsed_ms:.2f} ms")
        
        >>> # With lambda for parameterized calls
        >>> data, time_ms = measure_time(
        ...     lambda: process_data(input_data, threads=4),
        ...     name="parallel_processing"
        ... )
    """
    operation_name = name or (func.__name__ if hasattr(func, '__name__') else 'operation')
    func_logger = logger_instance or logger
    
    start_time = time.perf_counter()
    try:
        result = func()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        func_logger.log(level, f"{operation_name} completed in {elapsed_ms:.2f} ms")
        return result, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        func_logger.error(
            f"{operation_name} failed after {elapsed_ms:.2f} ms: "
            f"{type(e).__name__}: {str(e)}"
        )
        raise


# Helper function for consistent timing log formatting
def _log_timing(
    logger_instance: logging.Logger,
    level: int,
    operation_name: str,
    elapsed_ms: float,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    result: Any = None
) -> None:
    """
    Helper function for consistent timing log messages.
    
    Args:
        logger_instance: Logger to use
        level: Logging level
        operation_name: Name of the operation
        elapsed_ms: Elapsed time in milliseconds
        args: Optional function arguments to include
        kwargs: Optional function keyword arguments to include
        result: Optional function result to include
    """
    message_parts = [f"{operation_name}"]
    
    if args is not None or kwargs is not None:
        arg_parts = []
        if args:
            arg_parts.append(f"args={args}")
        if kwargs:
            arg_parts.append(f"kwargs={kwargs}")
        message_parts.append(f"({', '.join(arg_parts)})")
    
    message_parts.append(f" completed in {elapsed_ms:.2f} ms")
    
    if result is not None:
        result_str = str(result)
        if len(result_str) > 100:
            result_str = result_str[:100] + "..."
        message_parts.append(f" -> {result_str}")
    
    logger_instance.log(level, "".join(message_parts))