"""
Monitoring decorators for resource tracking and performance analysis.

This module provides decorators that can be applied to functions to automatically
monitor their resource usage, memory consumption, and performance characteristics.
"""

import functools
import gc
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from ..logging import get_logger
from .memory_tracking import log_memory_usage
from .system_metrics import log_system_stats

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def monitor_resources(
    threshold_memory_percent: float = 80.0,
    threshold_cpu_percent: float = 90.0,
    log_interval: float | None = None,
    alert_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[F], F]:
    """Monitor resource usage during function execution with optional thresholds."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = get_logger(func.__module__)

            # Import here to avoid circular imports
            from .performance_monitor import get_global_monitor

            monitor = get_global_monitor()

            # Start monitoring in a separate thread if interval is specified
            stop_monitoring = threading.Event()
            monitor_thread = None

            if log_interval:

                def periodic_monitor():
                    while not stop_monitoring.is_set():
                        stats = log_system_stats(logger_instance=func_logger)

                        # Check thresholds
                        if stats["memory"]["percent"] > threshold_memory_percent:
                            func_logger.warning(
                                f"High memory usage in {func.__name__}: "
                                f"{stats['memory']['percent']:.1f}% "
                                f"(threshold: {threshold_memory_percent}%)"
                            )
                            if alert_callback:
                                alert_callback(
                                    {
                                        "type": "memory",
                                        "function": func.__name__,
                                        "value": stats["memory"]["percent"],
                                        "threshold": threshold_memory_percent,
                                    }
                                )

                        if stats["cpu"]["total_percent"] > threshold_cpu_percent:
                            func_logger.warning(
                                f"High CPU usage in {func.__name__}: "
                                f"{stats['cpu']['total_percent']:.1f}% "
                                f"(threshold: {threshold_cpu_percent}%)"
                            )
                            if alert_callback:
                                alert_callback(
                                    {
                                        "type": "cpu",
                                        "function": func.__name__,
                                        "value": stats["cpu"]["total_percent"],
                                        "threshold": threshold_cpu_percent,
                                    }
                                )

                        stop_monitoring.wait(log_interval)

                monitor_thread = threading.Thread(target=periodic_monitor, daemon=True)
                monitor_thread.start()

            try:
                # Execute function with monitoring
                with monitor.monitor(func.__name__) as metrics:
                    result = func(*args, **kwargs)

                    # Final resource check
                    final_stats = log_system_stats(logger_instance=func_logger)

                    # Add to metrics
                    metrics.additional_metrics.update(
                        {
                            "final_memory_percent": final_stats["memory"]["percent"],
                            "final_cpu_percent": final_stats["cpu"]["total_percent"],
                            "exceeded_memory_threshold": final_stats["memory"][
                                "percent"
                            ]
                            > threshold_memory_percent,
                            "exceeded_cpu_threshold": final_stats["cpu"][
                                "total_percent"
                            ]
                            > threshold_cpu_percent,
                        }
                    )

                    return result
            finally:
                if monitor_thread:
                    stop_monitoring.set()
                    monitor_thread.join(timeout=1.0)

        return wrapper  # type: ignore

    return decorator


def profile_memory(
    track_allocations: bool = True, log_top_consumers: int = 10
) -> Callable[[F], F]:
    """Profile memory allocations during function execution."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = get_logger(func.__module__)

            # Log initial memory state
            initial_memory = log_memory_usage(
                f"Before {func.__name__}", logger_instance=func_logger
            )

            # Force garbage collection before starting
            gc.collect()

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Force garbage collection after completion
                gc.collect()

                # Log final memory state
                final_memory = log_memory_usage(
                    f"After {func.__name__}", logger_instance=func_logger
                )

                # Calculate changes
                memory_change = final_memory["rss_mb"] - initial_memory["rss_mb"]
                duration = time.time() - start_time

                func_logger.info(
                    f"{func.__name__} memory profile: "
                    f"Change={memory_change:+.2f}MB, "
                    f"Duration={duration:.3f}s, "
                    f"Rate={memory_change / duration if duration > 0 else 0:.2f}MB/s"
                )

                # Log garbage collection statistics
                if gc.isenabled():
                    gc_stats = gc.get_count()
                    func_logger.debug(f"GC counts after {func.__name__}: {gc_stats}")

        return wrapper  # type: ignore

    return decorator


def log_function_resources[F: Callable[..., Any]](func: F) -> F:
    """Simple decorator that logs resource usage for a function."""
    return monitor_resources(log_interval=None)(func)
