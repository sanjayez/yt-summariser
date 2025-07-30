"""Combined resilience patterns for comprehensive fault tolerance."""

from typing import Callable, TypeVar

from .circuit_breaker import circuit_breaker
from .retry import retry_with_backoff
from .timeout import timeout_handler
from ..exceptions import ExternalServiceError

F = TypeVar('F', bound=Callable[..., any])


def resilient_external_call(
    service_name: str,
    max_attempts: int = 3,
    timeout_seconds: float = 30.0,
    circuit_failure_threshold: int = 5
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        # Apply decorators in reverse order (timeout -> retry -> circuit breaker)
        decorated_func = timeout_handler(timeout_seconds)(func)
        decorated_func = retry_with_backoff(
            max_attempts=max_attempts,
            retry_exceptions=(ConnectionError, TimeoutError, ExternalServiceError)
        )(decorated_func)
        decorated_func = circuit_breaker(
            name=service_name,
            failure_threshold=circuit_failure_threshold,
            timeout=timeout_seconds
        )(decorated_func)
        
        return decorated_func
    
    return decorator


# Service-Specific Resilience Patterns

# YouTube API resilience
youtube_circuit_breaker = circuit_breaker(
    name="youtube_api",
    failure_threshold=5,
    recovery_timeout=120.0,
    timeout=30.0,
    expected_exceptions=(ExternalServiceError, ConnectionError, TimeoutError)
)

youtube_retry = retry_with_backoff(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retry_exceptions=(ConnectionError, TimeoutError),
    stop_exceptions=(ExternalServiceError,)
)

def youtube_resilient(func: F) -> F:
    return resilient_external_call(
        service_name="youtube_api",
        max_attempts=3,
        timeout_seconds=30.0,
        circuit_failure_threshold=5
    )(func)


# OpenAI API resilience
openai_circuit_breaker = circuit_breaker(
    name="openai_api",
    failure_threshold=3,
    recovery_timeout=180.0,
    timeout=60.0,
    expected_exceptions=(ExternalServiceError, ConnectionError, TimeoutError)
)

openai_retry = retry_with_backoff(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    retry_exceptions=(ConnectionError, TimeoutError),
    stop_exceptions=(ExternalServiceError,)
)

def openai_resilient(func: F) -> F:
    return resilient_external_call(
        service_name="openai_api",
        max_attempts=5,
        timeout_seconds=60.0,
        circuit_failure_threshold=3
    )(func)


# Vector store resilience
vector_store_circuit_breaker = circuit_breaker(
    name="vector_store",
    failure_threshold=3,
    recovery_timeout=60.0,
    timeout=30.0,
    expected_exceptions=(ConnectionError, TimeoutError)
)

vector_store_retry = retry_with_backoff(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    retry_exceptions=(ConnectionError, TimeoutError)
)

def vector_store_resilient(func: F) -> F:
    return resilient_external_call(
        service_name="vector_store",
        max_attempts=3,
        timeout_seconds=30.0,
        circuit_failure_threshold=3
    )(func)


# Additional service resilience patterns
def database_resilient(func: F) -> F:
    return resilient_external_call(
        service_name="database",
        max_attempts=3,
        timeout_seconds=15.0,
        circuit_failure_threshold=5
    )(func)


def web_scraping_resilient(func: F) -> F:
    return resilient_external_call(
        service_name="web_scraping",
        max_attempts=5,
        timeout_seconds=45.0,
        circuit_failure_threshold=7
    )(func)


def filesystem_resilient(func: F) -> F:
    # Only apply retry and timeout for filesystem operations
    decorated_func = timeout_handler(10.0)(func)
    decorated_func = retry_with_backoff(
        max_attempts=3,
        base_delay=0.1,
        max_delay=1.0,
        retry_exceptions=(OSError, IOError)
    )(decorated_func)
    
    return decorated_func


def cache_resilient(func: F) -> F:
    return resilient_external_call(
        service_name="cache",
        max_attempts=2,
        timeout_seconds=5.0,
        circuit_failure_threshold=3
    )(func)


def create_resilient_decorator(
    service_name: str,
    max_attempts: int = 3,
    timeout_seconds: float = 30.0,
    circuit_failure_threshold: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    recovery_timeout: float = 60.0
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        # Apply timeout
        decorated_func = timeout_handler(timeout_seconds)(func)
        
        # Apply retry with custom delays
        decorated_func = retry_with_backoff(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_exceptions=(ConnectionError, TimeoutError, ExternalServiceError)
        )(decorated_func)
        
        # Apply circuit breaker
        decorated_func = circuit_breaker(
            name=service_name,
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=recovery_timeout,
            timeout=timeout_seconds
        )(decorated_func)
        
        return decorated_func
    
    return decorator