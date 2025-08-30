"""
Timing utilities for performance measurement and profiling.

This module provides production-ready timing utilities including:
- Decorators for timing function execution (sync and async)
- Context managers for timing code blocks
- Integration with the RAGPerformanceTracker pattern
- Detailed logging with millisecond precision
"""

import asyncio
import functools
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    TypeVar,
    cast,
)

from core.utils.logging_utils import get_logger

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])

# Default logger for timing operations
logger = get_logger(__name__)


def timed_operation(
    name: str | None = None,
    logger_instance: logging.Logger | None = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
    threshold_ms: float | None = None,
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
                            func_logger,
                            level,
                            operation_name,
                            elapsed_ms,
                            args if include_args else None,
                            kwargs if include_args else None,
                            result if include_result else None,
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
                            func_logger,
                            level,
                            operation_name,
                            elapsed_ms,
                            args if include_args else None,
                            kwargs if include_args else None,
                            result if include_result else None,
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
        logger_instance: logging.Logger | None = None,
        level: int = logging.INFO,
        threshold_ms: float | None = None,
    ):
        self.name = name
        self.logger = logger_instance or logger
        self.level = level
        self.threshold_ms = threshold_ms
        self.start_time: float | None = None
        self.elapsed_ms: float | None = None

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
                    self.level, f"{self.name} completed in {self.elapsed_ms:.2f} ms"
                )


@asynccontextmanager
async def async_timing_context(
    name: str,
    logger_instance: logging.Logger | None = None,
    level: int = logging.INFO,
    threshold_ms: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
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
            f"{name} failed after {elapsed_ms:.2f} ms: {type(e).__name__}: {str(e)}"
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing_info["elapsed_ms"] = elapsed_ms

        if threshold_ms is None or elapsed_ms >= threshold_ms:
            func_logger.log(level, f"{name} completed in {elapsed_ms:.2f} ms")


def measure_time(
    func: Callable[[], Any],
    name: str | None = None,
    logger_instance: logging.Logger | None = None,
    level: int = logging.INFO,
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
    operation_name = name or (
        func.__name__ if hasattr(func, "__name__") else "operation"
    )
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


class PerformanceTimer:
    """
    Enhanced performance timer with RAGPerformanceTracker integration.

    Provides detailed timing analysis for multi-stage operations, compatible
    with the existing RAGPerformanceTracker pattern. Tracks individual stages
    and provides comprehensive performance summaries.

    This class extends the basic timing functionality to support the same
    performance analysis patterns used in the RAG pipeline.

    Attributes:
        name: Name of the overall operation
        logger: Logger instance for output
        timings: Dictionary of stage timings
        stage_order: List maintaining the order of stages
        start_time: Overall operation start time

    Example:
        >>> timer = PerformanceTimer("data_processing")
        >>>
        >>> with timer.time_stage("load_data"):
        ...     data = load_from_database()
        >>>
        >>> with timer.time_stage("transform"):
        ...     transformed = transform_data(data)
        >>>
        >>> with timer.time_stage("save_results"):
        ...     save_to_cache(transformed)
        >>>
        >>> timer.log_summary()  # Logs detailed performance breakdown
    """

    def __init__(
        self, name: str = "operation", logger_instance: logging.Logger | None = None
    ):
        self.name = name
        self.logger = logger_instance or logger
        self.timings: dict[str, float] = {}
        self.stage_order: list[str] = []
        self.start_time = time.perf_counter()

    @contextmanager
    def time_stage(self, stage_name: str) -> Iterator[None]:
        """
        Context manager for timing a specific stage.

        Compatible with RAGPerformanceTracker.time_stage pattern but for
        synchronous operations.

        Args:
            stage_name: Name of the stage to time

        Example:
            >>> with timer.time_stage("preprocessing"):
            ...     cleaned_data = preprocess(raw_data)
        """
        start = time.perf_counter()
        self.logger.debug(f"Starting stage: {stage_name}")

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timings[stage_name] = elapsed_ms
            self.stage_order.append(stage_name)
            self.logger.debug(f"Stage {stage_name}: {elapsed_ms:.2f} ms")

    @asynccontextmanager
    async def async_time_stage(self, stage_name: str) -> AsyncIterator[None]:
        """
        Async context manager for timing a specific stage.

        Direct equivalent of RAGPerformanceTracker.time_stage for async operations.

        Args:
            stage_name: Name of the stage to time

        Example:
            >>> async with timer.async_time_stage("api_call"):
            ...     response = await fetch_external_api()
        """
        start = time.perf_counter()
        self.logger.debug(f"Starting stage: {stage_name}")

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timings[stage_name] = elapsed_ms
            self.stage_order.append(stage_name)
            self.logger.debug(f"Stage {stage_name}: {elapsed_ms:.2f} ms")

    def add_stage_time(self, stage_name: str, elapsed_ms: float) -> None:
        """
        Manually add a stage timing.

        Useful when you already have timing information from elsewhere.

        Args:
            stage_name: Name of the stage
            elapsed_ms: Elapsed time in milliseconds
        """
        self.timings[stage_name] = elapsed_ms
        self.stage_order.append(stage_name)
        self.logger.debug(f"Stage {stage_name}: {elapsed_ms:.2f} ms")

    def get_stage_timings(self) -> dict[str, float]:
        """
        Get current stage timings.

        Returns a copy of the timings dictionary, compatible with
        RAGPerformanceTracker.get_stage_timings().

        Returns:
            Dictionary mapping stage names to elapsed time in milliseconds
        """
        return self.timings.copy()

    def log_summary(self, include_bottlenecks: bool = True) -> None:
        """
        Log comprehensive timing summary.

        Similar to RAGPerformanceTracker.log_request_summary but more generic.
        Provides detailed breakdown of all stages with percentages and
        performance assessment.

        Args:
            include_bottlenecks: Whether to identify and log bottlenecks
        """
        total_time_ms = (time.perf_counter() - self.start_time) * 1000
        total_measured_ms = sum(self.timings.values())

        self.logger.info(f"{self.name} timing summary:")
        self.logger.info(f"Total time: {total_time_ms:.2f} ms")

        if not self.timings:
            self.logger.info("No stages recorded")
            return

        # Sort stages by time descending
        sorted_stages = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        # Log each stage with percentage
        self.logger.info("Stage breakdown:")
        for stage, time_ms in sorted_stages:
            percentage = (
                (time_ms / total_measured_ms * 100) if total_measured_ms > 0 else 0
            )
            self.logger.info(f"  {stage}: {time_ms:.2f} ms ({percentage:.1f}%)")

        # Performance assessment
        if total_time_ms < 1000:
            status = "EXCELLENT"
        elif total_time_ms < 3000:
            status = "GOOD"
        elif total_time_ms < 5000:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS OPTIMIZATION"

        self.logger.info(f"Performance: {status} ({total_time_ms:.0f} ms)")

        # Bottleneck identification
        if include_bottlenecks and self.timings:
            max_stage, max_time = max(self.timings.items(), key=lambda x: x[1])
            max_percentage = (
                (max_time / total_measured_ms * 100) if total_measured_ms > 0 else 0
            )

            if max_percentage > 50:
                self.logger.info(
                    f"Primary bottleneck: {max_stage} "
                    f"({max_percentage:.1f}% of measured time)"
                )

        # Log overhead if significant
        overhead_ms = total_time_ms - total_measured_ms
        if overhead_ms > 100:  # More than 100ms overhead
            overhead_percentage = overhead_ms / total_time_ms * 100
            self.logger.info(
                f"Unmeasured overhead: {overhead_ms:.2f} ms "
                f"({overhead_percentage:.1f}% of total)"
            )

    def reset(self) -> None:
        """Reset all timings for reuse."""
        self.timings.clear()
        self.stage_order.clear()
        self.start_time = time.perf_counter()


# Helper function for consistent timing log formatting
def _log_timing(
    logger_instance: logging.Logger,
    level: int,
    operation_name: str,
    elapsed_ms: float,
    args: tuple | None = None,
    kwargs: dict | None = None,
    result: Any = None,
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


# Convenience exports for common use cases
__all__ = [
    "timed_operation",
    "TimingContext",
    "async_timing_context",
    "measure_time",
    "PerformanceTimer",
]
