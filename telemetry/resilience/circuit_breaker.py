"""Circuit breaker pattern implementation for preventing cascading failures."""

import asyncio
import functools
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, cast

from ..logging import get_logger
from ..exceptions import BaseYTSummarizerError

F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(BaseYTSummarizerError):
    def __init__(self, service_name: str, failure_count: int, 
                 last_failure_time: Optional[datetime] = None):
        message = f"Circuit breaker is OPEN for {service_name}"
        details = {
            "service": service_name,
            "failure_count": failure_count,
            "last_failure_time": last_failure_time.isoformat() if last_failure_time else None,
            "state": "OPEN"
        }
        super().__init__(message, details)


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: Optional[float] = 30.0
    expected_exceptions: tuple = (Exception,)


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized", extra={
            "circuit": name, "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        })
    
    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state
    
    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count
    
    @property
    def success_count(self) -> int:
        with self._lock:
            return self._success_count
    
    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return False
        time_since_failure = datetime.now() - self._last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._last_success_time = datetime.now()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")
    
    def _record_failure(self, exception: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            self._success_count = 0
            
            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' opened", extra={
                        "circuit": self.name, "state": "OPEN",
                        "failure_count": self._failure_count, "last_error": str(exception)
                    })
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed during half-open, reopening")
    
    def _can_execute(self) -> bool:
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                    return True
                return False
            elif self._state == CircuitBreakerState.HALF_OPEN:
                return True
            return False
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not self._can_execute():
            raise CircuitBreakerError(self.name, self._failure_count, self._last_failure_time)
        
        try:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            self._record_success()
            
            logger.debug(f"Circuit breaker '{self.name}' call succeeded", extra={
                "circuit": self.name, "execution_time": execution_time, "state": self._state.value
            })
            return result
            
        except self.config.expected_exceptions as e:
            self._record_failure(e)
            logger.warning(f"Circuit breaker '{self.name}' call failed", extra={
                "circuit": self.name, "error": str(e),
                "failure_count": self._failure_count, "state": self._state.value
            })
            raise
    
    async def acall(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        if not self._can_execute():
            raise CircuitBreakerError(self.name, self._failure_count, self._last_failure_time)
        
        try:
            start_time = time.perf_counter()
            
            if self.config.timeout:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = await func(*args, **kwargs)
            
            execution_time = time.perf_counter() - start_time
            self._record_success()
            
            logger.debug(f"Circuit breaker '{self.name}' async call succeeded", extra={
                "circuit": self.name, "execution_time": execution_time, "state": self._state.value
            })
            return result
            
        except (self.config.expected_exceptions, asyncio.TimeoutError) as e:
            self._record_failure(e)
            logger.warning(f"Circuit breaker '{self.name}' async call failed", extra={
                "circuit": self.name, "error": str(e),
                "failure_count": self._failure_count, "state": self._state.value
            })
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name, "state": self._state.value,
                "failure_count": self._failure_count, "success_count": self._success_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                }
            }


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: Optional[float] = 30.0,
    expected_exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exceptions=expected_exceptions
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


def reset_circuit_breaker(name: str) -> bool:
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
    with _cb_lock:
        count = 0
        for name in list(_circuit_breakers.keys()):
            if reset_circuit_breaker(name):
                count += 1
        logger.info(f"Reset {count} circuit breakers to CLOSED state")
        return count


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    with _cb_lock:
        return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}