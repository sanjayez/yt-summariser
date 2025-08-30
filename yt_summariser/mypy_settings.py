"""
Mypy-friendly Django settings

This settings file provides dummy/fallback values for environment variables
that Mypy needs to load Django settings without failing.

These values are ONLY used during type checking and are not used in actual
application runtime.
"""

import os

# Set dummy environment variables for Mypy before importing main settings
os.environ.setdefault("OPENAI_API_KEY", "dummy_key_for_mypy_type_checking")
os.environ.setdefault("PINECONE_API_KEY", "dummy_key_for_mypy_type_checking")
os.environ.setdefault("DECODO_AUTH_TOKEN", "dummy_token_for_mypy_type_checking")
os.environ.setdefault("DATABASE_URL", "sqlite:///mypy_dummy.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

# Import all settings from the main settings file
from yt_summariser.settings import *  # noqa: F403,F401
