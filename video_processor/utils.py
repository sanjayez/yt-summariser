import signal
import hashlib
import logging
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

# Timeout Context Manager
@contextmanager
def timeout(seconds, operation_name="Operation"):
    """Context manager for adding timeout to operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"{operation_name} timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Idempotency Key Generation
def generate_idempotency_key(task_name, *args, **kwargs):
    """Generate a unique idempotency key for task execution"""
    # Create a hash from task name and arguments
    content = f"{task_name}:{str(args)}:{str(sorted(kwargs.items()))}"
    return hashlib.md5(content.encode()).hexdigest()

def check_task_idempotency(task_name, *args, **kwargs):
    """Check if task has already been processed successfully"""
    key = f"task_idempotent:{generate_idempotency_key(task_name, *args, **kwargs)}"
    return cache.get(key)

def mark_task_complete(task_name, result, *args, **kwargs):
    """Mark task as completed with idempotency key"""
    key = f"task_idempotent:{generate_idempotency_key(task_name, *args, **kwargs)}"
    # Store for 24 hours
    cache.set(key, {'completed': True, 'result': result}, timeout=86400)

# Idempotency Decorator
def idempotent_task(func):
    """Decorator to make tasks idempotent"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        task_name = func.__name__
        
        # Check if already completed
        cached_result = check_task_idempotency(task_name, *args, **kwargs)
        if cached_result and cached_result.get('completed'):
            logger.info(f"Task {task_name} already completed, returning cached result")
            return cached_result.get('result')
        
        # Execute task
        result = func(self, *args, **kwargs)
        
        # Mark as completed
        mark_task_complete(task_name, result, *args, **kwargs)
        return result
    
    return wrapper

# Dead Letter Queue Handler
def handle_dead_letter_task(task_name, task_id, args, kwargs, exception):
    """Handle permanently failed tasks"""
    logger.critical(
        f"Task {task_name} permanently failed",
        extra={
            'task_id': task_id,
            'task_name': task_name,
            'task_args': args,  # Renamed to avoid LogRecord conflict
            'task_kwargs': kwargs,  # Renamed for consistency
            'exception': str(exception),
            'exception_type': type(exception).__name__
        }
    )
    
    # Store in dead letter queue (you could also send to monitoring system)
    dead_letter_key = f"dead_letter:{task_name}:{task_id}"
    dead_letter_data = {
        'task_name': task_name,
        'task_id': task_id,
        'task_args': args,  # Renamed for consistency
        'task_kwargs': kwargs,  # Renamed for consistency
        'exception': str(exception),
        'exception_type': type(exception).__name__,
        'timestamp': str(timezone.now())
    }
    
    # Store for 7 days for analysis
    cache.set(dead_letter_key, dead_letter_data, timeout=604800)

# Database Transaction with Callback
def atomic_with_callback(callback_func, *callback_args, **callback_kwargs):
    """Execute database operations atomically with a callback on commit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with transaction.atomic():
                result = func(*args, **kwargs)
                transaction.on_commit(
                    lambda: callback_func(*callback_args, **callback_kwargs)
                )
                return result
        return wrapper
    return decorator

# Progress Tracking
def update_task_progress(task_instance, step, progress_percent, meta=None):
    """Update task progress for monitoring"""
    # Skip progress updates if task doesn't have a valid ID (e.g., when called directly)
    if not hasattr(task_instance, 'request') or not task_instance.request.id:
        logger.debug(f"Skipping progress update for {step} - no valid task ID")
        return
    
    meta_data = {
        'step': step,
        'progress': progress_percent,
        **(meta or {})
    }
    
    try:
    task_instance.update_state(
        state='PROGRESS',
        meta=meta_data
    ) 
    except Exception as e:
        logger.warning(f"Failed to update task progress: {e}")
        # Don't raise the exception, just log it