"""
Telemetry package for YouTube Summarizer.
Provides centralized logging, timing, monitoring, error handling, and resilience patterns.

This package contains focused telemetry modules:
- telemetry.logging: Centralized logging utilities
- telemetry.timing: Performance timing and measurement
- telemetry.monitoring: System resource monitoring
- telemetry.exceptions: Error handling and custom exceptions
- telemetry.resilience: Circuit breakers, retry, and fault tolerance

For backward compatibility, all utilities are still available from telemetry.utils
"""

__version__ = "2.0.0"

# Import all utilities from focused modules for easy access
from .exceptions import *
from .logging import *
from .monitoring import *
from .resilience import *
from .timing import *
