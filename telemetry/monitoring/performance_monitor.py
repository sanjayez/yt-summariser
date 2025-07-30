"""
Performance monitoring system with centralized metrics collection.

This module provides the PerformanceMonitor class and related utilities for
collecting, recording, and analyzing performance metrics across the application.
"""

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from ..logging import get_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    start_time: float
    end_time: Optional[float] = None
    cpu_percent_start: float = 0.0
    cpu_percent_end: float = 0.0
    memory_mb_start: float = 0.0
    memory_mb_end: float = 0.0
    memory_percent_start: float = 0.0
    memory_percent_end: float = 0.0
    duration_seconds: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_duration(self) -> None:
        """Calculate duration if end_time is set."""
        if self.end_time:
            self.duration_seconds = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'duration_seconds': self.duration_seconds,
            'cpu_percent_change': self.cpu_percent_end - self.cpu_percent_start,
            'memory_mb_change': self.memory_mb_end - self.memory_mb_start,
            'memory_percent_change': self.memory_percent_end - self.memory_percent_start,
            'cpu_percent_final': self.cpu_percent_end,
            'memory_mb_final': self.memory_mb_end,
            'memory_percent_final': self.memory_percent_end,
            **self.additional_metrics
        }


class PerformanceMonitor:
    """Centralized performance monitoring system."""
    
    def __init__(self, name: str = "PerformanceMonitor", enable_gc_stats: bool = True):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        self.enable_gc_stats = enable_gc_stats
        self._metrics: Dict[str, List[PerformanceMetrics]] = {}
        self._lock = threading.Lock()
        
        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            self._baseline_memory = self._process.memory_info().rss / 1024 / 1024
        else:
            self._process = None
            self._baseline_memory = 0
            self.logger.warning("psutil not available, memory tracking disabled")
        
        self._start_time = time.time()
        
        if enable_gc_stats:
            gc.set_debug(gc.DEBUG_STATS)
    
    def record_metric(self, operation: str, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for an operation."""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(metrics)
        self.logger.debug(f"Recorded metrics for {operation}: {metrics.to_dict()}")
    
    def get_metrics_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for recorded metrics."""
        with self._lock:
            if operation:
                return self._calculate_summary(operation, self._metrics.get(operation, []))
            return {op: self._calculate_summary(op, metrics) for op, metrics in self._metrics.items()}
    
    def _calculate_summary(self, operation: str, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
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
                "total": sum(durations)
            },
            "memory_mb_change": {
                "min": min(memory_changes),
                "max": max(memory_changes),
                "avg": sum(memory_changes) / len(memory_changes)
            },
            "cpu_percent_change": {
                "min": min(cpu_changes),
                "max": max(cpu_changes),
                "avg": sum(cpu_changes) / len(cpu_changes)
            }
        }
    
    def log_summary(self, level: int = logging.INFO) -> None:
        """Log performance summary for all operations."""
        summary = self.get_metrics_summary()
        
        self.logger.log(level, "Performance Summary:")
        for operation, stats in summary.items():
            if stats['count'] > 0:
                self.logger.log(
                    level,
                    f"  {operation}: {stats['count']} calls, "
                    f"avg duration: {stats['duration']['avg']:.3f}s, "
                    f"avg memory change: {stats['memory_mb_change']['avg']:.2f}MB"
                )
    
    def reset_metrics(self, operation: Optional[str] = None) -> None:
        """Reset collected metrics."""
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
        if self._process:
            metrics = PerformanceMetrics(
                start_time=time.time(),
                cpu_percent_start=self._process.cpu_percent(),
                memory_mb_start=self._process.memory_info().rss / 1024 / 1024,
                memory_percent_start=self._process.memory_percent()
            )
        else:
            metrics = PerformanceMetrics(
                start_time=time.time(),
                cpu_percent_start=0,
                memory_mb_start=0,
                memory_percent_start=0
            )
        
        try:
            yield metrics
        finally:
            # Collect ending metrics
            metrics.end_time = time.time()
            if self._process:
                metrics.cpu_percent_end = self._process.cpu_percent()
                metrics.memory_mb_end = self._process.memory_info().rss / 1024 / 1024
                metrics.memory_percent_end = self._process.memory_percent()
            else:
                metrics.cpu_percent_end = 0
                metrics.memory_mb_end = 0
                metrics.memory_percent_end = 0
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