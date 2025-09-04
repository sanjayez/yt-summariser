"""
Query Processor Celery Tasks
Handles asynchronous processing of QueryRequest entries
"""

import asyncio

from celery import shared_task
from django.db import transaction

from telemetry.logging.logger import get_logger

from .models import QueryRequest
from .services.playlist_processor import extract_playlist_videos
from .services.query_processing import QueryProcessor

logger = get_logger(__name__)


@shared_task(
    bind=True,
    name="query_processor.process_query_request",
    soft_time_limit=300,  # 5 minutes
    time_limit=360,  # 6 minutes
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
)
def process_query_request(self, search_id: str):
    logger.info(f"🔄 Starting async processing for QueryRequest {search_id}")

    # Get pending QueryRequest
    try:
        query_request = QueryRequest.objects.get(search_id=search_id, status="pending")
        logger.info(f"🔀 Processing QueryRequest {search_id} content")
    except QueryRequest.DoesNotExist:
        logger.warning(f"⚠️  QueryRequest {search_id} not found or already processed")
        return {"status": "already_claimed", "search_id": search_id}

    try:
        # Heavy processing
        result = asyncio.run(_process_request(query_request))

        # Atomic completion
        _update_query_request(query_request, result)

        logger.info(
            f"✅ Completed QueryRequest {search_id} with status: {result['status']}"
        )
        return {
            "status": result["status"],
            "search_id": search_id,
            "total_videos": result.get("total_videos", 0),
        }

    except Exception as e:
        error_msg = f"Failed to process QueryRequest {search_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Phase 4: Atomic error handling (async-safe)
        try:
            error_result = {
                "status": "failed",
                "error": str(e),
                "video_urls": [],
                "total_videos": 0,
            }
            _update_query_request(query_request, error_result)
            logger.error(f"❌ Marked QueryRequest {search_id} as failed")
        except Exception as save_error:
            logger.error(f"Failed to save error state for {search_id}: {save_error}")

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60) from e


async def _process_request(query_request: QueryRequest) -> dict:
    """Process QueryRequest using appropriate processor based on request type."""
    logger.info(
        f"🔀 Processing {query_request.request_type} request {query_request.search_id}"
    )

    try:
        # TODO: replace with node side car
        if query_request.request_type == "topic":
            # Topic processing: LLM enhancement + YouTube search (async)
            processor = QueryProcessor()
            return await processor.process_query_request(query_request)

        elif query_request.request_type == "playlist":
            # Playlist processing: Extract video URLs from playlist (uses configurable limit)
            video_urls = await asyncio.to_thread(
                extract_playlist_videos, query_request.original_content
            )
            return {
                "status": "success",
                "video_urls": video_urls,
                "total_videos": len(video_urls),
            }

        elif query_request.request_type == "video":
            # Video processing: Content already has the video URL
            return {
                "status": "success",
                "video_urls": [query_request.original_content],
                "total_videos": 1,
            }

        else:
            raise ValueError(f"Unsupported request type: {query_request.request_type}")

    except Exception as e:
        logger.error(
            f"Processing failed for {query_request.search_id}: {e}", exc_info=True
        )
        return {
            "status": "failed",
            "error": str(e),
            "video_urls": [],
            "concepts": [],
            "enhanced_queries": [],
            "total_videos": 0,
        }


def _update_query_request(query_request: QueryRequest, result: dict):
    """Sync helper to update QueryRequest in database."""
    with transaction.atomic():
        # Re-fetch with lock to ensure we have latest state
        obj = QueryRequest.objects.select_for_update().get(pk=query_request.pk)

        # Update all fields from processing result
        obj.status = result.get("status", "success")
        obj.video_urls = result.get("video_urls", [])
        obj.concepts = result.get("concepts", [])
        obj.enhanced_queries = result.get("enhanced_queries", [])
        obj.intent_type = result.get(
            "intent_type"
        )  # None for video/playlist or failures
        obj.total_videos = result.get("total_videos", 0)

        # Handle error message
        error_msg = result.get("error", "")
        obj.error_message = error_msg[:1000] if error_msg else ""

        # Save with explicit fields for performance
        obj.save(
            update_fields=[
                "status",
                "video_urls",
                "concepts",
                "enhanced_queries",
                "intent_type",
                "total_videos",
                "error_message",
            ]
        )
