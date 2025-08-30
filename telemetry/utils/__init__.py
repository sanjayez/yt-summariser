"""
Backward compatibility wrapper for telemetry utilities.

⚠️  DEPRECATED: This module structure is deprecated as of v2.0.0
Please migrate to the new focused telemetry modules:

OLD (deprecated):
    from telemetry.utils import get_logger, timed_operation

NEW (recommended):
    from telemetry.logging import get_logger
    from telemetry.timing import timed_operation

Or use the new unified imports:
    from telemetry import get_logger, timed_operation

The utilities have been refactored into focused modules for better maintainability:
- telemetry.logging: Centralized logging utilities
- telemetry.timing: Performance timing and measurement
- telemetry.monitoring: System resource monitoring
- telemetry.exceptions: Error handling and custom exceptions
- telemetry.resilience: Circuit breakers, retry, and fault tolerance

This compatibility layer will be maintained but may be removed in future versions.
"""

# Re-export everything from the new focused modules for backward compatibility
from ..exceptions import (
    BaseYTSummarizerError,
    ExternalServiceError,
    ValidationError,
    VideoProcessingError,
    format_exception_chain,
    get_error_summary,
    handle_api_errors,
    handle_exceptions,
    handle_timeout,
    log_exception,
    retry_on_exception,
    safe_cleanup,
    with_error_context,
)
from ..logging import JSONFormatter, get_logger, log_function_call, setup_logging
from ..monitoring import (
    PerformanceMonitor,
    get_global_monitor,
    get_process_info,
    log_function_resources,
    log_memory_usage,
    log_system_stats,
    memory_tracker,
    monitor_resources,
    profile_memory,
)
from ..resilience import (
    CircuitBreakerError,
    circuit_breaker,
    resilient_external_call,
    retry_with_backoff,
    timeout_handler,
)
from ..timing import PerformanceTimer, TimingContext, measure_time, timed_operation

# For backward compatibility, also make everything available as before
__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "log_function_call",
    "JSONFormatter",
    # Timing
    "timed_operation",
    "TimingContext",
    "measure_time",
    "PerformanceTimer",
    # Monitoring
    "PerformanceMonitor",
    "log_memory_usage",
    "log_system_stats",
    "monitor_resources",
    "profile_memory",
    "memory_tracker",
    "log_function_resources",
    "get_process_info",
    "get_global_monitor",
    # Exception handling
    "BaseYTSummarizerError",
    "VideoProcessingError",
    "ExternalServiceError",
    "ValidationError",
    "handle_exceptions",
    "with_error_context",
    "handle_timeout",
    "retry_on_exception",
    "handle_api_errors",
    "log_exception",
    "format_exception_chain",
    "get_error_summary",
    "safe_cleanup",
    # Resilience
    "circuit_breaker",
    "retry_with_backoff",
    "timeout_handler",
    "resilient_external_call",
    "CircuitBreakerError",
]
