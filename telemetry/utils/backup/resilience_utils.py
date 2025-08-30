"""
Resilience utilities for the YT Summariser application.

This module provides comprehensive resilience patterns including:
- Circuit breaker pattern for preventing cascading failures
- Retry logic with exponential backoff and jitter
- Timeout handling for both sync and async operations
- Integration support for external services (YouTube API, OpenAI, vector stores)
"""

import asyncio
import builtins
import functools
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Optional,
    TypeVar,
    cast,
)

try:
    from .exception_utils import BaseYTSummarizerError, ExternalServiceError
    from .logging_utils import get_logger
except ImportError:
    # Fallback for direct module imports
    from core.utils.exception_utils import BaseYTSummarizerError, ExternalServiceError
    from core.utils.logging_utils import get_logger

# Type variables
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# Module logger
logger = get_logger(__name__)


# Circuit Breaker Implementation


class CircuitBreakerState(Enum):
    """States for the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is recovered


class CircuitBreakerError(BaseYTSummarizerError):
    """Exception raised when circuit breaker is open."""

    def __init__(
        self,
        service_name: str,
        failure_count: int,
        last_failure_time: datetime | None = None,
    ):
        """
        Initialize circuit breaker error.

        Args:
            service_name: Name of the service
            failure_count: Number of consecutive failures
            last_failure_time: Time of last failure
        """
        message = f"Circuit breaker is OPEN for {service_name}"
        details = {
            "service": service_name,
            "failure_count": failure_count,
            "last_failure_time": last_failure_time.isoformat()
            if last_failure_time
            else None,
            "state": "OPEN",
        }
        super().__init__(message, details)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successful calls needed to close from half-open
    timeout: float | None = 30.0  # Request timeout in seconds
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is open, requests fail fast
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the service/circuit
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._lock = threading.RLock()

        logger.info(
            f"Circuit breaker '{name}' initialized",
            extra={
                "circuit": name,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                },
            },
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count (in half-open state)."""
        with self._lock:
            return self._success_count

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from OPEN to HALF_OPEN."""
        if self._last_failure_time is None:
            return False

        time_since_failure = datetime.now() - self._last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._failure_count = 0
            self._last_success_time = datetime.now()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    # Enough successes, close the circuit
                    self._state = CircuitBreakerState.CLOSED
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' closed after recovery",
                        extra={"circuit": self.name, "state": "CLOSED"},
                    )

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            self._success_count = 0  # Reset success count

            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    # Too many failures, open the circuit
                    self._state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened",
                        extra={
                            "circuit": self.name,
                            "state": "OPEN",
                            "failure_count": self._failure_count,
                            "last_error": str(exception),
                        },
                    )
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Failed during half-open, go back to open
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' failed during half-open, reopening",
                    extra={"circuit": self.name, "state": "OPEN"},
                )

    def _can_execute(self) -> bool:
        """Check if request can be executed."""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    # Try to go to half-open
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' entering half-open state",
                        extra={"circuit": self.name, "state": "HALF_OPEN"},
                    )
                    return True
                return False
            elif self._state == CircuitBreakerState.HALF_OPEN:
                return True

            return False

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        if not self._can_execute():
            raise CircuitBreakerError(
                self.name, self._failure_count, self._last_failure_time
            )

        try:
            start_time = time.perf_counter()

            # Apply timeout if configured
            if self.config.timeout and not asyncio.iscoroutinefunction(func):
                # For sync functions, we can use signal-based timeout on Unix systems
                result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            execution_time = time.perf_counter() - start_time
            self._record_success()

            logger.debug(
                f"Circuit breaker '{self.name}' call succeeded",
                extra={
                    "circuit": self.name,
                    "execution_time": execution_time,
                    "state": self._state.value,
                },
            )

            return result

        except self.config.expected_exceptions as e:
            self._record_failure(e)
            logger.warning(
                f"Circuit breaker '{self.name}' call failed",
                extra={
                    "circuit": self.name,
                    "error": str(e),
                    "failure_count": self._failure_count,
                    "state": self._state.value,
                },
            )
            raise

    async def acall(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        if not self._can_execute():
            raise CircuitBreakerError(
                self.name, self._failure_count, self._last_failure_time
            )

        try:
            start_time = time.perf_counter()

            # Apply timeout if configured
            if self.config.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)

            execution_time = time.perf_counter() - start_time
            self._record_success()

            logger.debug(
                f"Circuit breaker '{self.name}' async call succeeded",
                extra={
                    "circuit": self.name,
                    "execution_time": execution_time,
                    "state": self._state.value,
                },
            )

            return result

        except (self.config.expected_exceptions, asyncio.TimeoutError) as e:
            self._record_failure(e)
            logger.warning(
                f"Circuit breaker '{self.name}' async call failed",
                extra={
                    "circuit": self.name,
                    "error": str(e),
                    "failure_count": self._failure_count,
                    "state": self._state.value,
                },
            )
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time.isoformat()
                if self._last_failure_time
                else None,
                "last_success_time": self._last_success_time.isoformat()
                if self._last_success_time
                else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                },
            }


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """
    Get or create a circuit breaker instance.

    Args:
        name: Circuit breaker name
        config: Configuration (only used for new instances)

    Returns:
        Circuit breaker instance
    """
    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def circuit_breaker(
    name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float | None = 30.0,
    expected_exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker pattern to a function.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before trying half-open
        success_threshold: Successful calls needed to close from half-open
        timeout: Request timeout in seconds
        expected_exceptions: Exceptions that count as failures

    Returns:
        Decorated function

    Example:
        >>> @circuit_breaker(name="youtube_api", failure_threshold=3)
        ... def fetch_video_info(video_id: str) -> dict:
        ...     # This function will be protected by circuit breaker
        ...     return youtube_api.get_video(video_id)

        >>> @circuit_breaker(name="openai_api", timeout=45.0)
        ... async def generate_summary(text: str) -> str:
        ...     return await openai_client.create_completion(text)
    """

    def decorator(func: F) -> F:
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exceptions=expected_exceptions,
        )

        cb = get_circuit_breaker(circuit_name, config)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await cb.acall(func, *args, **kwargs)

            return cast(F, async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return cb.call(func, *args, **kwargs)

            return cast(F, sync_wrapper)

    return decorator


# Retry Logic with Backoff


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_exceptions: tuple = (Exception,)
    stop_exceptions: tuple = ()  # Exceptions that should not be retried


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for retry attempt with exponential backoff and jitter.

    Args:
        attempt: Attempt number (0-based)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = min(config.base_delay * (config.backoff_factor**attempt), config.max_delay)

    # Add jitter to prevent thundering herd
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay

    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,),
    stop_exceptions: tuple = (),
    logger_instance: Optional = None,
) -> Callable[[F], F]:
    """
    Decorator to retry function execution with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Factor to multiply delay by after each failure
        jitter: Whether to add random jitter to delays
        retry_exceptions: Exception types to retry on
        stop_exceptions: Exception types that should stop retrying immediately
        logger_instance: Logger to use for retry messages

    Returns:
        Decorated function

    Example:
        >>> @retry_with_backoff(max_attempts=5, base_delay=2.0)
        ... def fetch_data(url: str) -> dict:
        ...     response = requests.get(url)
        ...     response.raise_for_status()
        ...     return response.json()

        >>> @retry_with_backoff(
        ...     max_attempts=3,
        ...     retry_exceptions=(ConnectionError, TimeoutError),
        ...     stop_exceptions=(ValueError,)
        ... )
        ... async def process_video(video_id: str) -> dict:
        ...     return await video_service.process(video_id)
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
        stop_exceptions=stop_exceptions,
    )

    def decorator(func: F) -> F:
        func_logger = logger_instance or get_logger(func.__module__)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None

                for attempt in range(config.max_attempts):
                    try:
                        result = await func(*args, **kwargs)

                        # Log successful retry
                        if attempt > 0:
                            func_logger.info(
                                f"{func.__name__} succeeded on attempt {attempt + 1}",
                                extra={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                },
                            )

                        return result

                    except config.stop_exceptions as e:
                        # Don't retry these exceptions
                        func_logger.error(
                            f"{func.__name__} failed with non-retryable error: {str(e)}",
                            extra={"function": func.__name__, "error": str(e)},
                        )
                        raise

                    except config.retry_exceptions as e:
                        last_exception = e

                        if attempt == config.max_attempts - 1:
                            # Last attempt, don't wait
                            func_logger.error(
                                f"{func.__name__} failed after {config.max_attempts} attempts",
                                extra={
                                    "function": func.__name__,
                                    "attempts": config.max_attempts,
                                    "final_error": str(e),
                                },
                            )
                            break

                        # Calculate delay and wait
                        delay = calculate_delay(attempt, config)
                        func_logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}, "
                            f"retrying in {delay:.2f}s: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts,
                                "delay": delay,
                                "error": str(e),
                            },
                        )

                        await asyncio.sleep(delay)

                # All attempts failed
                if last_exception:
                    raise last_exception

                # Should never reach here
                raise RuntimeError(f"{func.__name__} failed with unknown error")

            return cast(F, async_wrapper)

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None

                for attempt in range(config.max_attempts):
                    try:
                        result = func(*args, **kwargs)

                        # Log successful retry
                        if attempt > 0:
                            func_logger.info(
                                f"{func.__name__} succeeded on attempt {attempt + 1}",
                                extra={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                },
                            )

                        return result

                    except config.stop_exceptions as e:
                        # Don't retry these exceptions
                        func_logger.error(
                            f"{func.__name__} failed with non-retryable error: {str(e)}",
                            extra={"function": func.__name__, "error": str(e)},
                        )
                        raise

                    except config.retry_exceptions as e:
                        last_exception = e

                        if attempt == config.max_attempts - 1:
                            # Last attempt, don't wait
                            func_logger.error(
                                f"{func.__name__} failed after {config.max_attempts} attempts",
                                extra={
                                    "function": func.__name__,
                                    "attempts": config.max_attempts,
                                    "final_error": str(e),
                                },
                            )
                            break

                        # Calculate delay and wait
                        delay = calculate_delay(attempt, config)
                        func_logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}, "
                            f"retrying in {delay:.2f}s: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts,
                                "delay": delay,
                                "error": str(e),
                            },
                        )

                        time.sleep(delay)

                # All attempts failed
                if last_exception:
                    raise last_exception

                # Should never reach here
                raise RuntimeError(f"{func.__name__} failed with unknown error")

            return cast(F, sync_wrapper)

    return decorator


# Timeout Handling


class TimeoutError(BaseYTSummarizerError):
    """Exception raised when an operation times out."""

    def __init__(self, operation: str, timeout_seconds: float, **kwargs: Any):
        """
        Initialize timeout error.

        Args:
            operation: Name of the operation that timed out
            timeout_seconds: Timeout value in seconds
            **kwargs: Additional error details
        """
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        details.update(kwargs)
        super().__init__(message, details)


def timeout_handler(
    timeout_seconds: float,
    error_message: str | None = None,
    logger_instance: Optional = None,
) -> Callable[[F], F]:
    """
    Decorator to add timeout handling to functions.

    Args:
        timeout_seconds: Timeout in seconds
        error_message: Custom error message for timeout
        logger_instance: Logger to use for timeout messages

    Returns:
        Decorated function

    Example:
        >>> @timeout_handler(30.0, "Video processing timed out")
        ... async def process_video(video_id: str) -> dict:
        ...     # Long-running async operation
        ...     await some_long_operation()
        ...     return {"status": "processed"}

        >>> # For sync functions, timeout is less precise but still works
        >>> @timeout_handler(10.0)
        ... def fetch_data(url: str) -> dict:
        ...     return requests.get(url, timeout=10).json()
    """

    def decorator(func: F) -> F:
        func_logger = logger_instance or get_logger(func.__module__)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout_seconds
                    )
                except builtins.TimeoutError as e:
                    msg = (
                        error_message
                        or f"{func.__name__} timed out after {timeout_seconds}s"
                    )
                    func_logger.error(
                        msg,
                        extra={
                            "function": func.__name__,
                            "timeout": timeout_seconds,
                            "args": args,
                            "kwargs": kwargs,
                        },
                    )
                    raise TimeoutError(
                        func.__name__, timeout_seconds, args=args, kwargs=kwargs
                    ) from e

            return cast(F, async_wrapper)

        else:
            # For sync functions, we can't easily implement timeout without threads
            # But we can at least document the expected timeout
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.perf_counter() - start_time

                    # Log if execution time exceeds timeout (warning, not error)
                    if execution_time > timeout_seconds:
                        func_logger.warning(
                            f"{func.__name__} took {execution_time:.2f}s "
                            f"(exceeded expected timeout of {timeout_seconds}s)",
                            extra={
                                "function": func.__name__,
                                "execution_time": execution_time,
                                "timeout": timeout_seconds,
                            },
                        )

                    return result

                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    func_logger.error(
                        f"{func.__name__} failed after {execution_time:.2f}s",
                        extra={
                            "function": func.__name__,
                            "execution_time": execution_time,
                            "error": str(e),
                        },
                    )
                    raise

            return cast(F, sync_wrapper)

    return decorator


# Service-Specific Resilience Patterns

# YouTube API resilience
youtube_circuit_breaker = circuit_breaker(
    name="youtube_api",
    failure_threshold=5,
    recovery_timeout=120.0,  # 2 minutes
    timeout=30.0,
    expected_exceptions=(ExternalServiceError, ConnectionError, TimeoutError),
)

youtube_retry = retry_with_backoff(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retry_exceptions=(ConnectionError, TimeoutError),
    stop_exceptions=(ExternalServiceError,),  # Don't retry on auth/quota errors
)

# OpenAI API resilience
openai_circuit_breaker = circuit_breaker(
    name="openai_api",
    failure_threshold=3,
    recovery_timeout=180.0,  # 3 minutes
    timeout=60.0,
    expected_exceptions=(ExternalServiceError, ConnectionError, TimeoutError),
)

openai_retry = retry_with_backoff(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    retry_exceptions=(ConnectionError, TimeoutError),
    stop_exceptions=(ExternalServiceError,),  # Don't retry on API key/quota errors
)

# Vector store resilience
vector_store_circuit_breaker = circuit_breaker(
    name="vector_store",
    failure_threshold=3,
    recovery_timeout=60.0,
    timeout=30.0,
    expected_exceptions=(ConnectionError, TimeoutError),
)

vector_store_retry = retry_with_backoff(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    retry_exceptions=(ConnectionError, TimeoutError),
)


# Composite Decorators for Common Use Cases


def resilient_external_call(
    service_name: str,
    max_attempts: int = 3,
    timeout_seconds: float = 30.0,
    circuit_failure_threshold: int = 5,
) -> Callable[[F], F]:
    """
    Composite decorator that applies circuit breaker, retry, and timeout patterns.

    This is a convenience decorator for external service calls that need
    comprehensive resilience patterns.

    Args:
        service_name: Name of the external service
        max_attempts: Maximum retry attempts
        timeout_seconds: Request timeout
        circuit_failure_threshold: Circuit breaker failure threshold

    Returns:
        Decorated function with all resilience patterns applied

    Example:
        >>> @resilient_external_call("youtube_api", max_attempts=5, timeout_seconds=45.0)
        ... async def fetch_video_metadata(video_id: str) -> dict:
        ...     # This call is protected by circuit breaker, retry, and timeout
        ...     return await youtube_client.get_video(video_id)
    """

    def decorator(func: F) -> F:
        # Apply decorators in reverse order (timeout -> retry -> circuit breaker)
        decorated_func = timeout_handler(timeout_seconds)(func)
        decorated_func = retry_with_backoff(
            max_attempts=max_attempts,
            retry_exceptions=(ConnectionError, TimeoutError, ExternalServiceError),
        )(decorated_func)
        decorated_func = circuit_breaker(
            name=service_name,
            failure_threshold=circuit_failure_threshold,
            timeout=timeout_seconds,
        )(decorated_func)

        return decorated_func

    return decorator


# Utility Functions


def get_all_circuit_breaker_stats() -> dict[str, dict[str, Any]]:
    """
    Get statistics for all circuit breakers.

    Returns:
        Dictionary mapping circuit breaker names to their statistics
    """
    with _cb_lock:
        return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}


def reset_circuit_breaker(name: str) -> bool:
    """
    Reset a circuit breaker to CLOSED state.

    Args:
        name: Circuit breaker name

    Returns:
        True if reset successfully, False if circuit breaker not found
    """
    with _cb_lock:
        if name in _circuit_breakers:
            cb = _circuit_breakers[name]
            with cb._lock:
                cb._state = CircuitBreakerState.CLOSED
                cb._failure_count = 0
                cb._success_count = 0
                cb._last_failure_time = None
            logger.info(f"Circuit breaker '{name}' manually reset to CLOSED")
            return True
        return False


def reset_all_circuit_breakers() -> int:
    """
    Reset all circuit breakers to CLOSED state.

    Returns:
        Number of circuit breakers reset
    """
    with _cb_lock:
        count = 0
        for name in list(_circuit_breakers.keys()):
            if reset_circuit_breaker(name):
                count += 1
        logger.info(f"Reset {count} circuit breakers to CLOSED state")
        return count


# Export all public APIs
__all__ = [
    # Circuit breaker
    "CircuitBreakerState",
    "CircuitBreakerError",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "circuit_breaker",
    "get_circuit_breaker",
    # Retry logic
    "RetryConfig",
    "retry_with_backoff",
    "calculate_delay",
    # Timeout handling
    "TimeoutError",
    "timeout_handler",
    # Service-specific decorators
    "youtube_circuit_breaker",
    "youtube_retry",
    "openai_circuit_breaker",
    "openai_retry",
    "vector_store_circuit_breaker",
    "vector_store_retry",
    # Composite decorators
    "resilient_external_call",
    # Utility functions
    "get_all_circuit_breaker_stats",
    "reset_circuit_breaker",
    "reset_all_circuit_breakers",
]
