"""
Resilience utilities for the YT Summariser application.

This package provides comprehensive resilience patterns including:
- Circuit breaker pattern for preventing cascading failures
- Retry logic with exponential backoff and jitter
- Timeout handling for both sync and async operations
- Integration support for external services (YouTube API, OpenAI, vector stores)

The package is organized into focused modules:
- circuit_breaker: Circuit breaker pattern implementation
- retry: Retry logic with backoff functionality
- timeout: Timeout handling functionality
- composite: Combined resilience patterns for common use cases
"""

# Import all public APIs from submodules

# Circuit breaker functionality
from .circuit_breaker import (
    CircuitBreakerState,
    CircuitBreakerError,
    CircuitBreakerConfig,
    CircuitBreaker,
    circuit_breaker,
    get_circuit_breaker,
    reset_circuit_breaker,
    reset_all_circuit_breakers,
    get_all_circuit_breaker_stats
)

# Retry functionality
from .retry import (
    RetryConfig,
    retry_with_backoff,
    calculate_delay,
    retry_async,
    retry_sync
)

# Timeout functionality
from .timeout import (
    TimeoutError,
    timeout_handler,
    with_timeout,
    with_timeout_sync,
    TimeoutContext,
    create_timeout_wrapper,
    timeout_10s,
    timeout_30s,
    timeout_60s,
    timeout_5min,
    timeout_10min
)

# Composite resilience patterns
from .composite import (
    resilient_external_call,
    youtube_circuit_breaker,
    youtube_retry,
    youtube_resilient,
    openai_circuit_breaker,
    openai_retry,
    openai_resilient,
    vector_store_circuit_breaker,
    vector_store_retry,
    vector_store_resilient,
    database_resilient,
    web_scraping_resilient,
    filesystem_resilient,
    cache_resilient,
    create_resilient_decorator
)

# Export all public APIs for backward compatibility
__all__ = [
    # Circuit breaker
    'CircuitBreakerState',
    'CircuitBreakerError',
    'CircuitBreakerConfig',
    'CircuitBreaker',
    'circuit_breaker',
    'get_circuit_breaker',
    'reset_circuit_breaker',
    'reset_all_circuit_breakers',
    'get_all_circuit_breaker_stats',
    
    # Retry logic
    'RetryConfig',
    'retry_with_backoff',
    'calculate_delay',
    'retry_async',
    'retry_sync',
    
    # Timeout handling
    'TimeoutError',
    'timeout_handler',
    'with_timeout',
    'with_timeout_sync',
    'TimeoutContext',
    'create_timeout_wrapper',
    'timeout_10s',
    'timeout_30s',
    'timeout_60s',
    'timeout_5min',
    'timeout_10min',
    
    # Composite patterns
    'resilient_external_call',
    'youtube_circuit_breaker',
    'youtube_retry',
    'youtube_resilient',
    'openai_circuit_breaker',
    'openai_retry',
    'openai_resilient',
    'vector_store_circuit_breaker',
    'vector_store_retry',
    'vector_store_resilient',
    'database_resilient',
    'web_scraping_resilient',
    'filesystem_resilient',
    'cache_resilient',
    'create_resilient_decorator'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "YT Summariser Team"
__description__ = "Comprehensive resilience patterns for fault-tolerant applications"