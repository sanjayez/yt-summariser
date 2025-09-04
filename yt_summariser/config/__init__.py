"""
Configuration modules for YT Summariser.

Centralized configuration management for:
- Logging configuration
- Database settings
- Cache configuration
- External service settings
"""

from .logging import get_logging_config

__all__ = [
    "get_logging_config",
]
