"""
Performance monitoring utilities for the YT Summariser application.

This module provides comprehensive system and application monitoring capabilities:
- Memory usage tracking and profiling
- CPU utilization monitoring
- Resource monitoring decorators
- Performance metrics collection
- System statistics logging

All monitoring functions integrate with the application's logging system
for consistent output and analysis.
"""

import functools
import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

import psutil  # type: ignore[import-untyped]

from .logging_utils import get_logger

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])

# Default logger for monitoring
logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    start_time: float
    end_time: float | None = None
    cpu_percent_start: float = 0.0
    cpu_percent_end: float = 0.0
    memory_mb_start: float = 0.0
    memory_mb_end: float = 0.0
    memory_percent_start: float = 0.0
    memory_percent_end: float = 0.0
    duration_seconds: float = 0.0
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def calculate_duration(self) -> None:
        """Calculate duration if end_time is set."""
        if self.end_time:
            self.duration_seconds = self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "duration_seconds": self.duration_seconds,
            "cpu_percent_change": self.cpu_percent_end - self.cpu_percent_start,
            "memory_mb_change": self.memory_mb_end - self.memory_mb_start,
            "memory_percent_change": self.memory_percent_end
            - self.memory_percent_start,
            "cpu_percent_final": self.cpu_percent_end,
            "memory_mb_final": self.memory_mb_end,
            "memory_percent_final": self.memory_percent_end,
            **self.additional_metrics,
        }


class PerformanceMonitor:
    """
    Centralized performance monitoring system.

    Provides thread-safe collection and reporting of performance metrics
    across the application.
    """

    def __init__(self, name: str = "PerformanceMonitor", enable_gc_stats: bool = False):
        """
        Initialize the performance monitor.

        Args:
            name: Monitor instance name for logging
            enable_gc_stats: Whether to track garbage collection statistics
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        self.enable_gc_stats = enable_gc_stats
        self._metrics: dict[str, list[PerformanceMetrics]] = {}
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())

        # Initialize baseline metrics
        self._baseline_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        self._start_time = time.time()

        if enable_gc_stats:
            # Enable garbage collection statistics
            gc.set_debug(gc.DEBUG_STATS)

    def record_metric(self, operation: str, metrics: PerformanceMetrics) -> None:
        """
        Record performance metrics for an operation.

        Args:
            operation: Name of the operation being monitored
            metrics: Performance metrics to record
        """
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(metrics)

        self.logger.debug(f"Recorded metrics for {operation}: {metrics.to_dict()}")

    def get_metrics_summary(self, operation: str | None = None) -> dict[str, Any]:
        """
        Get summary statistics for recorded metrics.

        Args:
            operation: Specific operation to summarize (None for all)

        Returns:
            Dictionary containing summary statistics
        """
        with self._lock:
            if operation:
                metrics_list = self._metrics.get(operation, [])
                return self._calculate_summary(operation, metrics_list)
            else:
                summary = {}
                for op, metrics_list in self._metrics.items():
                    summary[op] = self._calculate_summary(op, metrics_list)
                return summary

    def _calculate_summary(
        self, operation: str, metrics_list: list[PerformanceMetrics]
    ) -> dict[str, Any]:
        """Calculate summary statistics for a list of metrics."""
        if not metrics_list:
            return {"operation": operation, "count": 0}

        durations = [m.duration_seconds for m in metrics_list]
        memory_changes = [m.memory_mb_end - m.memory_mb_start for m in metrics_list]
        cpu_changes = [m.cpu_percent_end - m.cpu_percent_start for m in metrics_list]

        return {
            "operation": operation,
            "count": len(metrics_list),
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations),
            },
            "memory_mb_change": {
                "min": min(memory_changes),
                "max": max(memory_changes),
                "avg": sum(memory_changes) / len(memory_changes),
            },
            "cpu_percent_change": {
                "min": min(cpu_changes),
                "max": max(cpu_changes),
                "avg": sum(cpu_changes) / len(cpu_changes),
            },
        }

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log performance summary for all operations."""
        summary = self.get_metrics_summary()

        self.logger.log(level, "Performance Summary:")
        for operation, stats in summary.items():
            if stats["count"] > 0:
                self.logger.log(
                    level,
                    f"  {operation}: {stats['count']} calls, "
                    f"avg duration: {stats['duration']['avg']:.3f}s, "
                    f"avg memory change: {stats['memory_mb_change']['avg']:.2f}MB",
                )

    def reset_metrics(self, operation: str | None = None) -> None:
        """
        Reset collected metrics.

        Args:
            operation: Specific operation to reset (None for all)
        """
        with self._lock:
            if operation:
                self._metrics.pop(operation, None)
            else:
                self._metrics.clear()

        self.logger.info(f"Reset metrics for: {operation or 'all operations'}")

    @contextmanager
    def monitor(self, operation: str, log_on_complete: bool = True):
        """
        Context manager for monitoring an operation.

        Args:
            operation: Name of the operation to monitor
            log_on_complete: Whether to log metrics when operation completes

        Yields:
            PerformanceMetrics instance that can be used to add custom metrics

        Example:
            >>> monitor = PerformanceMonitor()
            >>> with monitor.monitor("video_processing") as metrics:
            ...     # Do some work
            ...     metrics.additional_metrics['frames_processed'] = 1000
        """
        # Collect starting metrics
        metrics = PerformanceMetrics(
            start_time=time.time(),
            cpu_percent_start=self._process.cpu_percent(),
            memory_mb_start=self._process.memory_info().rss / 1024 / 1024,
            memory_percent_start=self._process.memory_percent(),
        )

        try:
            yield metrics
        finally:
            # Collect ending metrics
            metrics.end_time = time.time()
            metrics.cpu_percent_end = self._process.cpu_percent()
            metrics.memory_mb_end = self._process.memory_info().rss / 1024 / 1024
            metrics.memory_percent_end = self._process.memory_percent()
            metrics.calculate_duration()

            # Record the metrics
            self.record_metric(operation, metrics)

            if log_on_complete:
                self.logger.info(
                    f"{operation} completed in {metrics.duration_seconds:.3f}s "
                    f"(memory: {metrics.memory_mb_end - metrics.memory_mb_start:+.2f}MB, "
                    f"cpu: {metrics.cpu_percent_end - metrics.cpu_percent_start:+.1f}%)"
                )


# Global performance monitor instance
_global_monitor = PerformanceMonitor("GlobalMonitor")


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def log_memory_usage(
    label: str = "Current",
    include_gc_stats: bool = True,
    logger_instance: Any | None = None,
) -> dict[str, float]:
    """
    Log current memory usage statistics.

    Args:
        label: Label for the memory log entry
        include_gc_stats: Whether to include garbage collection statistics
        logger_instance: Logger to use (defaults to module logger)

    Returns:
        Dictionary containing memory statistics in MB

    Example:
        >>> stats = log_memory_usage("After processing")
        >>> # Logs: "After processing Memory Usage: RSS=150.23MB, VMS=512.45MB, ..."
    """
    log = logger_instance or logger
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Convert to MB
    stats = {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024,
    }

    # Add platform-specific metrics if available
    if hasattr(memory_info, "shared"):
        stats["shared_mb"] = memory_info.shared / 1024 / 1024
    if hasattr(memory_info, "pss"):
        stats["pss_mb"] = memory_info.pss / 1024 / 1024

    log.info(
        f"{label} Memory Usage: "
        f"RSS={stats['rss_mb']:.2f}MB, "
        f"VMS={stats['vms_mb']:.2f}MB, "
        f"Percent={stats['percent']:.1f}%, "
        f"Available={stats['available_mb']:.2f}MB"
    )

    if include_gc_stats and gc.isenabled():
        gc_stats = gc.get_stats()
        if gc_stats:
            # Log generation statistics
            for i, gen_stats in enumerate(gc_stats[:3]):  # Only first 3 generations
                log.debug(
                    f"  GC Generation {i}: "
                    f"collections={gen_stats.get('collections', 0)}, "
                    f"collected={gen_stats.get('collected', 0)}, "
                    f"uncollectable={gen_stats.get('uncollectable', 0)}"
                )

    return stats


def log_system_stats(
    logger_instance: Any | None = None,
    include_disk: bool = True,
    include_network: bool = False,
) -> dict[str, Any]:
    """
    Log comprehensive system statistics.

    Args:
        logger_instance: Logger to use (defaults to module logger)
        include_disk: Whether to include disk I/O statistics
        include_network: Whether to include network I/O statistics

    Returns:
        Dictionary containing system statistics

    Example:
        >>> stats = log_system_stats()
        >>> # Logs CPU, memory, and optionally disk/network statistics
    """
    log = logger_instance or logger
    stats = {}

    # CPU statistics
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    stats["cpu"] = {
        "total_percent": sum(cpu_percent) / len(cpu_percent),
        "per_core": cpu_percent,
        "count": psutil.cpu_count(),
        "count_physical": psutil.cpu_count(logical=False),
    }

    # Memory statistics
    virtual_memory = psutil.virtual_memory()
    stats["memory"] = {
        "total_mb": virtual_memory.total / 1024 / 1024,
        "available_mb": virtual_memory.available / 1024 / 1024,
        "used_mb": virtual_memory.used / 1024 / 1024,
        "percent": virtual_memory.percent,
    }

    # Process-specific statistics
    process = psutil.Process(os.getpid())
    stats["process"] = {
        "num_threads": process.num_threads(),
        "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
    }

    log.info(
        f"System Stats: "
        f"CPU={stats['cpu']['total_percent']:.1f}% "
        f"({stats['cpu']['count']} cores), "
        f"Memory={stats['memory']['percent']:.1f}% "
        f"({stats['memory']['used_mb']:.0f}/{stats['memory']['total_mb']:.0f}MB), "
        f"Process: {stats['process']['num_threads']} threads, "
        f"{stats['process']['memory_mb']:.0f}MB"
    )

    # Disk I/O statistics
    if include_disk:
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                stats["disk_io"] = {
                    "read_mb": disk_io.read_bytes / 1024 / 1024,
                    "write_mb": disk_io.write_bytes / 1024 / 1024,
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count,
                }
                log.debug(
                    f"  Disk I/O: "
                    f"Read={stats['disk_io']['read_mb']:.2f}MB, "
                    f"Write={stats['disk_io']['write_mb']:.2f}MB"
                )
        except Exception as e:
            log.debug(f"Could not collect disk I/O stats: {e}")

    # Network I/O statistics
    if include_network:
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                stats["network_io"] = {
                    "sent_mb": net_io.bytes_sent / 1024 / 1024,
                    "recv_mb": net_io.bytes_recv / 1024 / 1024,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                log.debug(
                    f"  Network I/O: "
                    f"Sent={stats['network_io']['sent_mb']:.2f}MB, "
                    f"Received={stats['network_io']['recv_mb']:.2f}MB"
                )
        except Exception as e:
            log.debug(f"Could not collect network I/O stats: {e}")

    return stats


def monitor_resources(
    threshold_memory_percent: float = 80.0,
    threshold_cpu_percent: float = 90.0,
    log_interval: float | None = None,
    alert_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to monitor resource usage during function execution.

    Monitors CPU and memory usage, logging warnings if thresholds are exceeded.

    Args:
        threshold_memory_percent: Memory usage threshold for warnings (default: 80%)
        threshold_cpu_percent: CPU usage threshold for warnings (default: 90%)
        log_interval: Interval for periodic logging during execution (seconds)
        alert_callback: Optional callback when thresholds are exceeded

    Returns:
        Decorated function

    Example:
        >>> @monitor_resources(threshold_memory_percent=70.0)
        ... def process_large_dataset(data):
        ...     # Process data
        ...     return results

        >>> # If memory usage exceeds 70%, a warning will be logged
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = get_logger(func.__module__)
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
    """
    Decorator to profile memory allocations during function execution.

    Args:
        track_allocations: Whether to track individual allocations
        log_top_consumers: Number of top memory consumers to log

    Returns:
        Decorated function

    Example:
        >>> @profile_memory(log_top_consumers=5)
        ... def analyze_data(dataset):
        ...     # Memory-intensive operations
        ...     return analysis_results
    """

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


@contextmanager
def memory_tracker(name: str, logger_instance: Any | None = None):
    """
    Context manager for tracking memory usage within a code block.

    Args:
        name: Name for the tracked operation
        logger_instance: Logger to use (defaults to module logger)

    Yields:
        Dictionary that will contain memory statistics after the block

    Example:
        >>> with memory_tracker("data_processing") as mem_stats:
        ...     # Process large dataset
        ...     processed_data = transform_data(raw_data)
        ... print(f"Memory used: {mem_stats['memory_change_mb']}MB")
    """
    log = logger_instance or logger
    process = psutil.Process(os.getpid())

    # Initial measurements
    gc.collect()  # Clean up before measuring
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    # Stats dictionary to be populated
    stats: dict[str, Any] = {}

    try:
        yield stats
    finally:
        # Final measurements
        gc.collect()  # Clean up before final measurement
        end_memory = process.memory_info().rss / 1024 / 1024
        end_time = time.time()

        # Populate statistics
        stats.update(
            {
                "name": name,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "memory_change_mb": end_memory - start_memory,
                "duration_seconds": end_time - start_time,
                "memory_percent": process.memory_percent(),
            }
        )

        log.info(
            f"Memory tracker '{name}': "
            f"Used {stats['memory_change_mb']:+.2f}MB in {stats['duration_seconds']:.3f}s "
            f"(Current: {stats['end_memory_mb']:.2f}MB, {stats['memory_percent']:.1f}%)"
        )


# Convenience functions for common monitoring tasks
def log_function_resources[F: Callable[..., Any]](func: F) -> F:
    """
    Simple decorator that logs resource usage for a function.

    Combines execution time and memory tracking with minimal overhead.

    Example:
        >>> @log_function_resources
        ... def compute_result(data):
        ...     return expensive_computation(data)
    """
    return monitor_resources(log_interval=None)(func)


def get_process_info() -> dict[str, Any]:
    """
    Get detailed information about the current process.

    Returns:
        Dictionary containing process information
    """
    process = psutil.Process(os.getpid())

    info = {
        "pid": process.pid,
        "name": process.name(),
        "status": process.status(),
        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
        "num_threads": process.num_threads(),
        "memory_info": {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
        },
        "cpu_times": {
            "user": process.cpu_times().user,
            "system": process.cpu_times().system,
        },
    }

    # Add platform-specific information
    try:
        info["num_fds"] = process.num_fds()
    except AttributeError:
        info["num_handles"] = (
            process.num_handles() if hasattr(process, "num_handles") else None
        )

    try:
        info["connections"] = len(process.connections())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        info["connections"] = None

    return info
