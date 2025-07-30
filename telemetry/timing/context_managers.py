"""
Timing context managers for performance measurement.

This module provides context managers for timing code blocks with detailed
logging and support for both synchronous and asynchronous contexts.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional
import logging

from ..logging import get_logger

# Default logger for timing operations
logger = get_logger(__name__)


class TimingContext:
    """
    Context manager for timing code blocks with detailed logging.
    
    Provides a clean way to time arbitrary code sections and automatically
    logs the elapsed time. Handles exceptions gracefully and logs them.
    
    Attributes:
        name: Name of the operation being timed
        logger: Logger instance for output
        level: Logging level for timing messages
        threshold_ms: Only log if execution exceeds this threshold
        start_time: Start time of the operation
        elapsed_ms: Elapsed time in milliseconds (available after exit)
    
    Example:
        >>> with TimingContext("database_query") as timer:
        ...     results = db.execute("SELECT * FROM users")
        ...     print(f"Query took {timer.elapsed_ms:.2f} ms")
        
        >>> # Can also be used with custom logger and threshold
        >>> with TimingContext("cache_lookup", logger=custom_logger, threshold_ms=10):
        ...     value = cache.get(key)
    """
    
    def __init__(
        self,
        name: str,
        logger_instance: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        threshold_ms: Optional[float] = None
    ):
        self.name = name
        self.logger = logger_instance or logger
        self.level = level
        self.threshold_ms = threshold_ms
        self.start_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None
    
    def __enter__(self) -> "TimingContext":
        """Start timing when entering the context."""
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and log results when exiting the context."""
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            
            if exc_type is not None:
                self.logger.error(
                    f"{self.name} failed after {self.elapsed_ms:.2f} ms: "
                    f"{exc_type.__name__}: {str(exc_val)}"
                )
            elif self.threshold_ms is None or self.elapsed_ms >= self.threshold_ms:
                self.logger.log(
                    self.level,
                    f"{self.name} completed in {self.elapsed_ms:.2f} ms"
                )


@asynccontextmanager
async def async_timing_context(
    name: str,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    threshold_ms: Optional[float] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async context manager for timing asynchronous code blocks.
    
    Similar to TimingContext but for async code. Provides timing information
    through a dictionary that's updated when the context exits.
    
    Args:
        name: Name of the operation being timed
        logger_instance: Logger to use (defaults to module logger)
        level: Logging level for timing messages
        threshold_ms: Only log if execution exceeds this threshold
    
    Yields:
        Dictionary with timing information (updated on exit)
    
    Example:
        >>> async with async_timing_context("api_request") as timing:
        ...     response = await client.get("/api/data")
        ...     # After context exits, timing["elapsed_ms"] is available
        ...     print(f"Request took {timing['elapsed_ms']:.2f} ms")
    """
    func_logger = logger_instance or logger
    start_time = time.perf_counter()
    timing_info = {"start_time": start_time, "elapsed_ms": None}
    
    func_logger.log(level, f"Starting {name}")
    
    try:
        yield timing_info
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing_info["elapsed_ms"] = elapsed_ms
        func_logger.error(
            f"{name} failed after {elapsed_ms:.2f} ms: "
            f"{type(e).__name__}: {str(e)}"
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing_info["elapsed_ms"] = elapsed_ms
        
        if threshold_ms is None or elapsed_ms >= threshold_ms:
            func_logger.log(level, f"{name} completed in {elapsed_ms:.2f} ms")