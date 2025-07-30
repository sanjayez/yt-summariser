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
from ..logging import get_logger, setup_logging, log_function_call, JSONFormatter
from ..timing import timed_operation, TimingContext, measure_time, PerformanceTimer
from ..monitoring import (
    PerformanceMonitor, log_memory_usage, log_system_stats, monitor_resources,
    profile_memory, memory_tracker, log_function_resources, get_process_info, get_global_monitor
)
from ..exceptions import (
    BaseYTSummarizerError, VideoProcessingError, ExternalServiceError, ValidationError,
    handle_exceptions, with_error_context, handle_timeout, retry_on_exception,
    handle_api_errors, log_exception, format_exception_chain, get_error_summary, safe_cleanup
)
from ..resilience import (
    circuit_breaker, retry_with_backoff, timeout_handler, resilient_external_call, CircuitBreakerError
)

# For backward compatibility, also make everything available as before
__all__ = [
    # Logging
    'get_logger',
    'setup_logging', 
    'log_function_call',
    'JSONFormatter',
    
    # Timing
    'timed_operation',
    'TimingContext',
    'measure_time',
    'PerformanceTimer',
    
    # Monitoring
    'PerformanceMonitor',
    'log_memory_usage',
    'log_system_stats',
    'monitor_resources',
    'profile_memory',
    'memory_tracker',
    'log_function_resources',
    'get_process_info',
    'get_global_monitor',
    
    # Exception handling
    'BaseYTSummarizerError',
    'VideoProcessingError',
    'ExternalServiceError',
    'ValidationError',
    'handle_exceptions',
    'with_error_context',
    'handle_timeout',
    'retry_on_exception',
    'handle_api_errors',
    'log_exception',
    'format_exception_chain',
    'get_error_summary',
    'safe_cleanup',
    
    # Resilience
    'circuit_breaker',
    'retry_with_backoff',
    'timeout_handler',
    'resilient_external_call',
    'CircuitBreakerError',
]