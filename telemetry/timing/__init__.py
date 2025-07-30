"""
Timing utilities for performance measurement and profiling.

This module provides production-ready timing utilities including:
- Decorators for timing function execution (sync and async)
- Context managers for timing code blocks
- Integration with the RAGPerformanceTracker pattern
- Detailed logging with millisecond precision

The module has been refactored into focused components:
- decorators: @timed_operation decorator and measure_time function
- context_managers: TimingContext and async_timing_context
- performance_timer: PerformanceTimer class for multi-stage timing

All functionality is re-exported here to maintain backward compatibility.
"""

# Import all functionality from focused modules
from .decorators import (
    timed_operation,
    measure_time,
)

from .context_managers import (
    TimingContext,
    async_timing_context,
)

from .performance_timer import (
    PerformanceTimer,
)

# Convenience exports for common use cases - maintaining original API
__all__ = [
    "timed_operation",
    "TimingContext", 
    "async_timing_context",
    "measure_time",
    "PerformanceTimer",
]