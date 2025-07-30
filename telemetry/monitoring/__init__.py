"""
Performance monitoring utilities for the YT Summariser application.

This package provides comprehensive system and application monitoring capabilities:
- Memory usage tracking and profiling
- CPU utilization monitoring
- Resource monitoring decorators
- Performance metrics collection
- System statistics logging

All monitoring functions integrate with the application's logging system
for consistent output and analysis.
"""

# Import all public classes and functions to maintain the same API
from .decorators import (
    log_function_resources,
    monitor_resources,
    profile_memory,
)
from .memory_tracking import (
    log_memory_usage,
    memory_tracker,
)
from .performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    get_global_monitor,
)
from .system_metrics import (
    get_process_info,
    log_system_stats,
)

# Expose all public functionality
__all__ = [
    # Performance monitoring classes
    'PerformanceMetrics',
    'PerformanceMonitor',
    'get_global_monitor',
    
    # Memory tracking functions
    'log_memory_usage',
    'memory_tracker',
    
    # System metrics functions
    'log_system_stats',
    'get_process_info',
    
    # Decorators
    'monitor_resources',
    'profile_memory',
    'log_function_resources',
]