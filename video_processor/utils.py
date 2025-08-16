import hashlib
import logging
import json
import time as _time
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
from typing import Optional

import redis
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from api.models import URLRequestTable

logger = logging.getLogger(__name__)

# Timeout Context Manager - Thread-safe implementation
@contextmanager
def timeout(seconds, operation_name="Operation"):
    """Context manager for adding timeout to operations - thread-safe for Celery"""
    import threading
    import time
    
    # For Celery workers, we'll use a simple timer approach
    # This is less precise but works across threads
    start_time = time.time()
    
    class TimeoutChecker:
        def __init__(self, timeout_seconds, op_name):
            self.timeout_seconds = timeout_seconds
            self.op_name = op_name
            self.start_time = time.time()
        
        def check(self):
            if time.time() - self.start_time > self.timeout_seconds:
                raise TimeoutError(f"{self.op_name} timed out after {self.timeout_seconds} seconds")
    
    checker = TimeoutChecker(seconds, operation_name)
    try:
        yield checker
    except Exception as e:
        # Log timeout issues but don't fail the entire task
        if "timed out" in str(e):
            logger.warning(f"Timeout occurred in {operation_name}: {e}")
        raise

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

    # ----------------------------------------------------------------------------------
    # Minimal SSE publishing for single-video progress via Redis pub/sub
    # Channel: video.{request_uuid}.progress
    # Messages mirror topic stream schema: stage_start, stage_progress, stage_complete, complete, error
    # ----------------------------------------------------------------------------------
    try:
        # Feature flag (default True for MVP); allow turning off via settings
        publish_enabled = getattr(settings, 'VIDEO_SSE_PUBLISH', True)
        if not publish_enabled:
            return

        request_uuid = _resolve_request_uuid(task_instance)
        if not request_uuid:
            return

        message_type, payload = _build_video_progress_payload(step, progress_percent, meta or {})
        if not message_type:
            return

        channel = f"video.{request_uuid}.progress"
        _publish_to_redis(channel, {
            'type': message_type,
            'request_id': str(request_uuid),
            'timestamp': _time.time(),
            **payload
        })
    except Exception as pub_error:
        # Never fail the task due to progress pub issues
        logger.debug(f"Non-fatal: failed to publish video progress update: {pub_error}")


def _resolve_request_uuid(task_instance) -> Optional[str]:
    """Try to resolve URLRequestTable.request_id (UUID string) from Celery task args/kwargs."""
    try:
        url_request_id = None
        # Prefer kwargs
        if hasattr(task_instance.request, 'kwargs') and task_instance.request.kwargs:
            url_request_id = task_instance.request.kwargs.get('url_request_id')
        # Fallback: scan args (often last positional arg)
        if url_request_id is None and hasattr(task_instance.request, 'args'):
            for arg in reversed(task_instance.request.args or []):
                if isinstance(arg, (int, str)):  # Accept UUID string or other formats
                    url_request_id = arg
                    break
        if url_request_id is None:
            return None
        # Convert to string if needed (request_id is now always UUID)
        return str(url_request_id)
    except Exception:
        return None


def _build_video_progress_payload(step: str, progress: int | float, meta: dict) -> tuple[Optional[str], dict]:
    """Map internal step/progress to topic-like SSE payload."""
    step_l = (step or '').lower()

    # Terminal completion
    if 'completed' in step_l or step_l == 'completed':
        return 'complete', {'message': 'Processing completed'}

    # Determine stage name
    stage = None
    if 'metadata' in step_l:
        stage = 'METADATA'
    elif 'transcript' in step_l:
        stage = 'TRANSCRIPT'
    elif 'summary' in step_l:
        stage = 'SUMMARY'
    elif 'embed' in step_l:
        stage = 'EMBEDDING'
    elif 'status' in step_l:
        stage = 'STATUS'
    elif 'analyz' in step_l:
        stage = 'ANALYZING'

    if not stage:
        # Unknown stage; skip publish
        return None, {}

    # Choose message type based on progress
    try:
        p = float(progress)
    except Exception:
        p = 0.0

    if p <= 0:
        msg_type = 'stage_start'
    elif p >= 100:
        msg_type = 'stage_complete'
    else:
        msg_type = 'stage_progress'

    payload = {
        'stage': stage,
        'message': meta.get('description') or meta.get('message') or stage.capitalize(),
        'progress': round(p, 1),
        'stage_progress': round(p, 1)
    }
    return msg_type, payload


def _publish_to_redis(channel: str, payload: dict) -> None:
    """Publish JSON payload to Redis pub/sub on DB 3 (like topic explorer)."""
    host = getattr(settings, 'REDIS_HOST', 'localhost')
    port = getattr(settings, 'REDIS_PORT', 6379)
    client = redis.Redis(host=host, port=port, db=3, decode_responses=True)
    # Will raise on connection issues; caller wraps
    client.publish(channel, json.dumps(payload))