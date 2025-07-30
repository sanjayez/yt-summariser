"""
Exception handling utilities for the YT Summariser application.

DEPRECATED: This module has been refactored into focused modules in core.exceptions.
All imports are now re-exported from the new location for backward compatibility.

New structure:
- core.exceptions.custom_exceptions: All custom exception classes
- core.exceptions.handlers: Exception handling decorators and context managers
- core.exceptions.context: Error context, logging, and utility functions
- core.exceptions.retry: Retry functionality

Please update imports to use 'from core.exceptions import ...' instead.
"""

# Import everything from the new modular structure to maintain backward compatibility
from core.exceptions import *

# Re-export the __all__ for backward compatibility
from core.exceptions import __all__