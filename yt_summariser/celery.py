import contextlib
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yt_summariser.settings")

app = Celery("yt_summariser")
app.config_from_object("django.conf:settings", namespace="CELERY")

# Additional Celery configuration for task state tracking
app.conf.update(
    # Enable comprehensive task state tracking
    task_track_started=True,
    task_send_sent_event=True,
    send_events=True,
    worker_send_task_events=True,
    # Result backend reliability settings
    result_backend_always_retry=True,
    result_backend_max_retries=10,
)

app.autodiscover_tasks()

# Non-blocking Gemini warmup when worker starts (best-effort)
try:
    import threading as _threading
    import time
    from typing import Dict

    from celery import signals

    from temp_file_logger import append_line

    @signals.worker_ready.connect
    def _warmup_gemini_llm(**kwargs):
        try:
            import asyncio

            from ai_utils.services.registry import (
                warmup_gemini_llm,
                warmup_vector_store,
            )

            def _warm():
                try:
                    asyncio.run(warmup_gemini_llm())
                    asyncio.run(warmup_vector_store())
                except Exception:
                    pass

            _threading.Thread(target=_warm, daemon=True).start()
        except Exception:
            pass

    # Lightweight per-task runtime logging for video_processor tasks
    _task_start_times: dict[str, float] = {}
    _task_lock = _threading.Lock()

    def _extract_request_id(args, kwargs):
        # Try kwargs first
        if isinstance(kwargs, dict):
            rid = kwargs.get("url_request_id")
            if isinstance(rid, int | str):
                return rid
        # Fallback: last positional arg often carries url_request_id in our pipeline
        try:
            if args:
                candidate = args[-1]
                if isinstance(candidate, int | str):
                    return candidate
        except Exception:
            pass
        return "unknown"

    def _preview(value, max_len: int = 200):
        try:
            if isinstance(value, dict):
                return {"keys": list(value.keys())[:10]}
            if isinstance(value, list | tuple):
                return {"type": type(value).__name__, "len": len(value)}
            if isinstance(value, str):
                return value[:max_len] + ("..." if len(value) > max_len else "")
            if isinstance(value, int | float | bool):
                return value
            return str(type(value).__name__)
        except Exception:
            return "unpreviewable"

    @signals.task_prerun.connect
    def _task_prerun_handler(
        sender=None, task_id=None, task=None, args=None, kwargs=None, **extra
    ):
        try:
            name = getattr(task, "name", "") or str(sender)
            if not name.startswith("video_processor."):
                return
            if name in (
                "video_processor.log_stage_duration",
                "video_processor.log_pipeline_total",
            ):
                return
            ts = time.time()
            with _task_lock:
                _task_start_times[task_id] = ts
            # Best-effort input preview
            rid = _extract_request_id(args or (), kwargs or {})
            with contextlib.suppress(Exception):
                append_line(
                    file_path=os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
                    message=f"task_start stage={name} request_id={rid} task_id={task_id} args_preview={_preview(args)} kwargs_keys={(list(kwargs.keys()) if isinstance(kwargs, dict) else [])}",
                )
        except Exception:
            pass

    @signals.task_postrun.connect
    def _task_postrun_handler(
        sender=None,
        task_id=None,
        task=None,
        args=None,
        kwargs=None,
        retval=None,
        **extra,
    ):
        try:
            name = getattr(task, "name", "") or str(sender)
            if not name.startswith("video_processor."):
                return
            if name in (
                "video_processor.log_stage_duration",
                "video_processor.log_pipeline_total",
            ):
                return
            with _task_lock:
                start_ts = _task_start_times.pop(task_id, None)
            if start_ts is None:
                return
            elapsed = max(0.0, time.time() - float(start_ts))
            req_id = _extract_request_id(args or (), kwargs or {})
            # Append runtime and a light retval preview
            retval_preview = _preview(retval)
            append_line(
                file_path=os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
                message=f"task_end stage={name} request_id={req_id} elapsed_s={elapsed:.2f} task_id={task_id} retval={retval_preview}",
            )
        except Exception:
            pass

    @signals.task_failure.connect
    def _task_failure_handler(
        sender=None,
        task_id=None,
        exception=None,
        args=None,
        kwargs=None,
        einfo=None,
        **extra,
    ):
        try:
            name = str(sender)
            if not name.startswith("video_processor."):
                return
            req_id = _extract_request_id(args or (), kwargs or {})
            exc_type = type(exception).__name__ if exception else "UnknownError"
            msg = str(exception)[:500] if exception else ""
            append_line(
                file_path=os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
                message=f"task_error stage={name} request_id={req_id} task_id={task_id} error_type={exc_type} error={msg}",
            )
        except Exception:
            pass
except Exception:
    pass
