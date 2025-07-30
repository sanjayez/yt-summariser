"""
Performance timer for multi-stage timing analysis.

This module provides the PerformanceTimer class for detailed timing analysis
of multi-stage operations, compatible with RAGPerformanceTracker patterns.
"""

import time
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncIterator, Dict, Iterator, Optional
import logging

from ..logging import get_logger

# Default logger for timing operations
logger = get_logger(__name__)


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
        self,
        name: str = "operation",
        logger_instance: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger_instance or logger
        self.timings: Dict[str, float] = {}
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
    
    def get_stage_timings(self) -> Dict[str, float]:
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
        sorted_stages = sorted(
            self.timings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Log each stage with percentage
        self.logger.info("Stage breakdown:")
        for stage, time_ms in sorted_stages:
            percentage = (time_ms / total_measured_ms * 100) if total_measured_ms > 0 else 0
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
            max_percentage = (max_time / total_measured_ms * 100) if total_measured_ms > 0 else 0
            
            if max_percentage > 50:
                self.logger.info(
                    f"Primary bottleneck: {max_stage} "
                    f"({max_percentage:.1f}% of measured time)"
                )
        
        # Log overhead if significant
        overhead_ms = total_time_ms - total_measured_ms
        if overhead_ms > 100:  # More than 100ms overhead
            overhead_percentage = (overhead_ms / total_time_ms * 100)
            self.logger.info(
                f"Unmeasured overhead: {overhead_ms:.2f} ms "
                f"({overhead_percentage:.1f}% of total)"
            )
    
    def reset(self) -> None:
        """Reset all timings for reuse."""
        self.timings.clear()
        self.stage_order.clear()
        self.start_time = time.perf_counter()