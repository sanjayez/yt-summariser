"""
API Services Module

This module contains business logic services for the API layer.
Services are organized by domain and responsibility following SOLID principles.

Available services:
- CacheService: Embedding and response caching
- ResponseService: Response formatting and compression
- SearchService: RAG search operations and coordination
- ServiceContainer: Dependency injection and service management
"""

from .cache_service import CacheService
from .response_service import ResponseService
from .search_service import SearchService
from .service_container import ServiceContainer

__all__ = [
    "CacheService",
    "ResponseService",
    "SearchService",
    "ServiceContainer",
]
