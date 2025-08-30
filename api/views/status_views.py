"""
Status Views - Handles real-time video processing status streaming via Redis.
Contains views for Server-Sent Events (SSE) status updates using Redis pub/sub only.
"""

import contextlib
import json
import time
from collections.abc import AsyncGenerator
from uuid import UUID

from django.conf import settings
from django.http import HttpRequest, StreamingHttpResponse
from django.views.decorators.http import require_http_methods

from telemetry import get_logger, handle_exceptions
from video_processor.config import API_CONFIG

try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    AsyncRedis = None


logger = get_logger(__name__)


@require_http_methods(["GET"])
@handle_exceptions(reraise=True)
async def video_status_stream(
    request: HttpRequest, request_id: UUID
) -> StreamingHttpResponse:
    """
    Stream real-time status updates for video processing.

    This endpoint provides Server-Sent Events (SSE) streaming for:
    1. Overall processing status
    2. Individual stage completion status
    3. Progress percentage calculation
    4. Detailed stage information

    The streaming includes all processing stages:
    - Metadata extraction
    - Transcript extraction
    - Summary generation
    - Content embedding

    Args:
        request: HTTP request
        request_id: UUID of the video processing request

    Returns:
        StreamingHttpResponse with text/event-stream content
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events for video processing status.

        Uses Redis pub/sub to stream updates until a terminal event is received
        or a timeout occurs.

        Yields:
            str: Formatted SSE data strings
        """
        logger.info(f"Starting status stream for request: {request_id}")

        # Send initial comment and a sane client reconnect delay to avoid rapid reconnect loops
        yield f": connected {request_id}\n\n"
        yield "retry: 3000\n\n"
        # Padding to defeat proxy/buffer delays and force immediate flush
        yield f": padding {' ' * 4096}\n\n"

        # Stream strictly via Redis pub/sub (async)
        if AsyncRedis is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Redis client not available'})}\n\n"
            logger.error("Redis client not available; cannot stream status updates")
            return

        try:
            host = getattr(settings, "REDIS_HOST", "localhost")
            port = getattr(settings, "REDIS_PORT", 6379)
            password = getattr(settings, "REDIS_PASSWORD", None)

            redis_config = {
                "host": host,
                "port": port,
                "db": 3,
                "decode_responses": True,
            }

            if password:
                redis_config["password"] = password

            redis_client = AsyncRedis(**redis_config)
            await redis_client.ping()

            pubsub = redis_client.pubsub()
            channel = f"video.{request_id}.progress"
            await pubsub.subscribe(channel)

            # Connected message and padding
            yield f"data: {json.dumps({'type': 'connected', 'message': '\ud83d\ude80 Real-time progress tracking active', 'mode': 'redis_pubsub'})}\n\n"
            yield f": padding to prevent buffering {' ' * 2048}\n\n"

            start_ts = time.time()
            max_duration = API_CONFIG["SSE"].get("event_timeout", 120)
            keepalive_interval = API_CONFIG["SSE"].get("keepalive_interval", 30)
            last_heartbeat_ts = time.time()
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is not None and message.get("type") == "message":
                    data = message.get("data")
                    if data:
                        yield f"data: {data}\n\n"
                        last_heartbeat_ts = time.time()
                        # Check terminal events
                        try:
                            parsed = json.loads(data)
                            if parsed.get("type") in ["complete", "error"]:
                                break
                        except Exception:
                            pass

                # Periodic heartbeat to keep the stream flushing even when quiet
                if time.time() - last_heartbeat_ts >= keepalive_interval:
                    yield ": heartbeat\n\n"
                    last_heartbeat_ts = time.time()

                if time.time() - start_ts > max_duration:
                    yield f"data: {json.dumps({'type': 'timeout', 'message': 'Stream timeout reached'})}\n\n"
                    break

            with contextlib.suppress(Exception):
                await pubsub.close()
            with contextlib.suppress(Exception):
                await redis_client.aclose()
        except Exception as exc:
            logger.error(f"Redis streaming error for {request_id}: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

    # Create SSE response with proper headers
    response = StreamingHttpResponse(
        event_stream(), content_type="text/event-stream; charset=utf-8"
    )
    response["Cache-Control"] = "no-cache, no-transform"
    response["X-Accel-Buffering"] = "no"  # Disable nginx buffering
    response["Connection"] = "keep-alive"

    return response
