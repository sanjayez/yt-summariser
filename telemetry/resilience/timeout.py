"""Timeout handling functionality for both sync and async operations."""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, TypeVar, cast

from ..logging import get_logger
from ..exceptions import BaseYTSummarizerError

F = TypeVar('F', bound=Callable[..., Any])
logger = get_logger(__name__)


class TimeoutError(BaseYTSummarizerError):
    def __init__(self, operation: str, timeout_seconds: float, **kwargs: Any):
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        details.update(kwargs)
        super().__init__(message, details)


def timeout_handler(
    timeout_seconds: float,
    error_message: Optional[str] = None,
    logger_instance: Optional = None
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        func_logger = logger_instance or get_logger(func.__module__)
        
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
                    func_logger.error(msg, extra={
                        "function": func.__name__,
                        "timeout": timeout_seconds,
                        "args": args,
                        "kwargs": kwargs
                    })
                    raise TimeoutError(
                        func.__name__,
                        timeout_seconds,
                        args=args,
                        kwargs=kwargs
                    ) from e
            
            return cast(F, async_wrapper)
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.perf_counter() - start_time
                    
                    if execution_time > timeout_seconds:
                        func_logger.warning(f"{func.__name__} took {execution_time:.2f}s "
                                          f"(exceeded expected timeout of {timeout_seconds}s)")
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    func_logger.error(f"{func.__name__} failed after {execution_time:.2f}s", 
                                    extra={"function": func.__name__, "execution_time": execution_time, "error": str(e)})
                    raise
            
            return cast(F, sync_wrapper)
    
    return decorator


async def with_timeout(
    coro_or_func: Any,
    timeout_seconds: float,
    *args: Any,
    error_message: Optional[str] = None,
    **kwargs: Any
) -> Any:
    try:
        if asyncio.iscoroutine(coro_or_func):
            return await asyncio.wait_for(coro_or_func, timeout=timeout_seconds)
        else:
            return await asyncio.wait_for(
                coro_or_func(*args, **kwargs),
                timeout=timeout_seconds
            )
    except asyncio.TimeoutError as e:
        operation_name = getattr(coro_or_func, '__name__', str(coro_or_func))
        msg = error_message or f"Operation '{operation_name}' timed out after {timeout_seconds}s"
        logger.error(msg, extra={"operation": operation_name, "timeout": timeout_seconds, "args": args, "kwargs": kwargs})
        raise TimeoutError(operation_name, timeout_seconds, args=args, kwargs=kwargs) from e


def with_timeout_sync(
    func: Callable[..., Any],
    timeout_seconds: float,
    *args: Any,
    error_message: Optional[str] = None,
    **kwargs: Any
) -> Any:
    start_time = time.perf_counter()
    
    try:
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        if execution_time > timeout_seconds:
            msg = error_message or f"{func.__name__} took {execution_time:.2f}s (exceeded expected timeout of {timeout_seconds}s)"
            logger.warning(msg, extra={"function": func.__name__, "execution_time": execution_time, "timeout": timeout_seconds})
        
        return result
        
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        logger.error(f"{func.__name__} failed after {execution_time:.2f}s", 
                    extra={"function": func.__name__, "execution_time": execution_time, "error": str(e)})
        raise


class TimeoutContext:
    def __init__(self, timeout_seconds: float, operation_name: str = "operation"):
        self.timeout_seconds = timeout_seconds
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        logger.debug(f"Starting {self.operation_name} with {self.timeout_seconds}s timeout")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = time.perf_counter() - self.start_time
            
            if exc_type is None:
                if execution_time > self.timeout_seconds:
                    logger.warning(f"{self.operation_name} took {execution_time:.2f}s "
                                  f"(exceeded expected timeout of {self.timeout_seconds}s)")
                else:
                    logger.debug(f"{self.operation_name} completed in {execution_time:.2f}s")
            else:
                logger.error(f"{self.operation_name} failed after {execution_time:.2f}s: {exc_val}")
        
        return False


def create_timeout_wrapper(timeout_seconds: float) -> Callable[[Callable], Callable]:
    return timeout_handler(timeout_seconds)


# Pre-configured timeout decorators for common use cases
timeout_10s = timeout_handler(10.0)
timeout_30s = timeout_handler(30.0)
timeout_60s = timeout_handler(60.0)
timeout_5min = timeout_handler(300.0)
timeout_10min = timeout_handler(600.0)