"""Retry logic with exponential backoff and jitter functionality."""

import asyncio
import functools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, cast

from ..logging import get_logger

F = TypeVar('F', bound=Callable[..., Any])
logger = get_logger(__name__)


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_exceptions: tuple = (Exception,)
    stop_exceptions: tuple = ()


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    delay = min(config.base_delay * (config.backoff_factor ** attempt), config.max_delay)
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,),
    stop_exceptions: tuple = (),
    logger_instance: Optional = None
) -> Callable[[F], F]:
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
        stop_exceptions=stop_exceptions
    )
    
    def decorator(func: F) -> F:
        func_logger = logger_instance or get_logger(func.__module__)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None
                
                for attempt in range(config.max_attempts):
                    try:
                        result = await func(*args, **kwargs)
                        if attempt > 0:
                            func_logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                        return result
                        
                    except config.stop_exceptions as e:
                        func_logger.error(f"{func.__name__} failed with non-retryable error: {str(e)}")
                        raise
                        
                    except config.retry_exceptions as e:
                        last_exception = e
                        
                        if attempt == config.max_attempts - 1:
                            func_logger.error(f"{func.__name__} failed after {config.max_attempts} attempts")
                            break
                        
                        delay = calculate_delay(attempt, config)
                        func_logger.warning(f"{func.__name__} failed on attempt {attempt + 1}, "
                                          f"retrying in {delay:.2f}s: {str(e)}")
                        await asyncio.sleep(delay)
                
                if last_exception:
                    raise last_exception
                raise RuntimeError(f"{func.__name__} failed with unknown error")
            
            return cast(F, async_wrapper)
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None
                
                for attempt in range(config.max_attempts):
                    try:
                        result = func(*args, **kwargs)
                        if attempt > 0:
                            func_logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                        return result
                        
                    except config.stop_exceptions as e:
                        func_logger.error(f"{func.__name__} failed with non-retryable error: {str(e)}")
                        raise
                        
                    except config.retry_exceptions as e:
                        last_exception = e
                        
                        if attempt == config.max_attempts - 1:
                            func_logger.error(f"{func.__name__} failed after {config.max_attempts} attempts")
                            break
                        
                        delay = calculate_delay(attempt, config)
                        func_logger.warning(f"{func.__name__} failed on attempt {attempt + 1}, "
                                          f"retrying in {delay:.2f}s: {str(e)}")
                        time.sleep(delay)
                
                if last_exception:
                    raise last_exception
                raise RuntimeError(f"{func.__name__} failed with unknown error")
            
            return cast(F, sync_wrapper)
    
    return decorator


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,),
    stop_exceptions: tuple = (),
    **kwargs: Any
) -> Any:
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
        stop_exceptions=stop_exceptions
    )
    
    func_logger = get_logger(func.__module__)
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                func_logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
            return result
            
        except config.stop_exceptions as e:
            func_logger.error(f"{func.__name__} failed with non-retryable error: {str(e)}")
            raise
            
        except config.retry_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts - 1:
                func_logger.error(f"{func.__name__} failed after {config.max_attempts} attempts")
                break
            
            delay = calculate_delay(attempt, config)
            func_logger.warning(f"{func.__name__} failed on attempt {attempt + 1}, "
                              f"retrying in {delay:.2f}s: {str(e)}")
            await asyncio.sleep(delay)
    
    if last_exception:
        raise last_exception
    raise RuntimeError(f"{func.__name__} failed with unknown error")


def retry_sync(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,),
    stop_exceptions: tuple = (),
    **kwargs: Any
) -> Any:
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
        stop_exceptions=stop_exceptions
    )
    
    func_logger = get_logger(func.__module__)
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                func_logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
            return result
            
        except config.stop_exceptions as e:
            func_logger.error(f"{func.__name__} failed with non-retryable error: {str(e)}")
            raise
            
        except config.retry_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts - 1:
                func_logger.error(f"{func.__name__} failed after {config.max_attempts} attempts")
                break
            
            delay = calculate_delay(attempt, config)
            func_logger.warning(f"{func.__name__} failed on attempt {attempt + 1}, "
                              f"retrying in {delay:.2f}s: {str(e)}")
            time.sleep(delay)
    
    if last_exception:
        raise last_exception
    raise RuntimeError(f"{func.__name__} failed with unknown error")