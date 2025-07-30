"""
Function logging decorators for the YT Summariser application.

This module provides decorators for logging function calls and execution times:
- log_function_call: Logs function calls with arguments and return values
- log_execution_time: Logs function execution time
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


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
    
    Example:
        >>> @log_execution_time()
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        
        >>> # This will log:
        >>> # INFO - slow_function executed in 1.001 seconds
    """
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