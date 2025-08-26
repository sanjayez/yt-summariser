"""
Backward compatibility wrapper for core utilities.

⚠️  DEPRECATED: This module structure is deprecated as of v2.0.0
Please migrate to the new focused modules:

OLD (deprecated):
    from core.utils import get_logger, timed_operation

NEW (recommended):
    from core.logging import get_logger
    from core.timing import timed_operation
    
Or use the new unified imports:
    from core import get_logger, timed_operation

The utilities have been refactored into focused modules for better maintainability:
- core.logging: Centralized logging utilities
- core.timing: Performance timing and measurement  
- core.monitoring: System resource monitoring
- core.exceptions: Error handling and custom exceptions
- core.resilience: Circuit breakers, retry, and fault tolerance

This compatibility layer will be maintained but may be removed in future versions.
"""

# Re-export everything from the new focused modules for backward compatibility
from ...logging import *
from ...timing import *
from ...monitoring import *
from ...exceptions import *
from ...resilience import *

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