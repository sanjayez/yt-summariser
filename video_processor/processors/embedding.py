"""
Video Content Embedding Task
Implements 4-layer embedding strategy for comprehensive video content search:
1. Metadata embedding (video-level context)
2. Summary embedding (key insights)
3. Transcript chunks embedding (comprehensive search)
4. Segment embeddings (precise timestamp navigation)
"""

import asyncio
import logging
from typing import Any

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from celery_progress.backend import ProgressRecorder
from django.db import transaction

from ai_utils.models import VectorDocument, VectorQuery
from ai_utils.services.registry import get_vector_service
from api.models import URLRequestTable

from ..config import TASK_STATES, YOUTUBE_CONFIG
from ..models import VideoMetadata, VideoTranscript
from ..text_utils.chunking import (
    chunk_transcript_text,
    format_metadata_for_embedding,
    format_segment_for_embedding,
    format_summary_for_embedding,
    prepare_batch_embeddings,
    validate_embedding_text,
)
from ..utils import (
    handle_dead_letter_task,
    idempotent_task,
    update_task_progress,
)

logger = logging.getLogger(__name__)


def embed_video_content_sync(
    video_metadata: VideoMetadata, transcript: VideoTranscript
) -> dict[str, Any]:
    """
    Synchronous helper function to embed all video content using 4-layer strategy.
    Returns embedding statistics and results.
    """
    try:
        # Initialize services with native Weaviate vectorization
        from ai_utils.config import get_config

        config = get_config()
        vector_service = get_vector_service()
        # No embedding service needed - Weaviate handles embeddings natively

        video_id = video_metadata.video_id
        embedding_items = []

        # Layer 1: Metadata embedding
        logger.info(f"Preparing metadata embedding for video {video_id}")
        metadata_text = format_metadata_for_embedding(video_metadata)
        metadata_text = validate_embedding_text(metadata_text)

        embedding_items.append(
            {
                "id": f"meta_{video_id}",
                "text": metadata_text,
                "metadata": {
                    "type": "metadata",
                    "video_id": video_id,
                    "title": video_metadata.title or "Unknown",
                    "channel": video_metadata.channel_name or "Unknown",
                    "duration": video_metadata.duration or 0,
                },
            }
        )

        # Layer 2: Summary embedding (only if summary exists)
        if transcript.summary and transcript.summary.strip():
            logger.info(f"Preparing summary embedding for video {video_id}")
            summary_text = format_summary_for_embedding(
                transcript.summary, transcript.key_points
            )
            summary_text = validate_embedding_text(summary_text)

            embedding_items.append(
                {
                    "id": f"summary_{video_id}",
                    "text": summary_text,
                    "metadata": {
                        "type": "summary",
                        "video_id": video_id,
                        "title": video_metadata.title or "Unknown",
                        "key_points_count": len(transcript.key_points)
                        if transcript.key_points
                        else 0,
                    },
                }
            )

        # Layer 3: Transcript chunks embedding
        if transcript.transcript_text and transcript.transcript_text.strip():
            logger.info(f"Preparing transcript chunks embedding for video {video_id}")
            transcript_chunks = chunk_transcript_text(
                transcript.transcript_text, chunk_size=1000, chunk_overlap=200
            )

            for i, chunk in enumerate(transcript_chunks):
                chunk_text = validate_embedding_text(chunk)
                embedding_items.append(
                    {
                        "id": f"transcript_{video_id}_{i}",
                        "text": chunk_text,
                        "metadata": {
                            "type": "transcript_chunk",
                            "video_id": video_id,
                            "chunk_index": i,
                            "title": video_metadata.title or "Unknown",
                        },
                    }
                )

        # Layer 4: Segment embeddings
        segments = transcript.segments.all()
        if segments.exists():
            logger.info(
                f"Preparing segment embeddings for video {video_id}: {segments.count()} segments"
            )

            for segment in segments:
                segment_text = format_segment_for_embedding(segment)
                segment_text = validate_embedding_text(segment_text)

                embedding_items.append(
                    {
                        "id": segment.segment_id,
                        "text": segment_text,
                        "metadata": {
                            "type": "segment",
                            "video_id": video_id,
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "sequence_number": segment.sequence_number,
                            "title": video_metadata.title or "Unknown",
                            "youtube_url": segment.get_youtube_url_with_timestamp(),
                        },
                    }
                )

        logger.info(f"Prepared {len(embedding_items)} items for embedding")

        # Batch process using native Weaviate vectorization (use provider-configured batch size)
        batches = prepare_batch_embeddings(
            embedding_items, batch_size=config.weaviate.batch_size or 100
        )
        total_embedded = 0
        failed_embeddings = []

        async def _process_one_batch(
            batch_idx: int, batch: list[dict[str, Any]]
        ) -> int:
            try:
                logger.info(
                    f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} items"
                )
                vector_documents = [
                    VectorDocument(
                        id=item["id"],
                        text=item["text"],
                        embedding=None,
                        metadata=item["metadata"],
                    )
                    for item in batch
                ]
                result = await vector_service.upsert_documents(
                    documents=vector_documents,
                    job_id=f"embed_batch_{video_id}_{batch_idx}",
                )
                return int(result.get("upserted_count", 0))
            except Exception as embed_error:
                logger.error(f"Error processing batch {batch_idx + 1}: {embed_error}")
                failed_embeddings.extend([item["id"] for item in batch])
                return 0

        # Run batches concurrently with a small cap (2–4)
        concurrency = min(4, max(2, (config.max_concurrent_requests or 4)))
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded_process(idx: int, b: list[dict[str, Any]]):
            async with semaphore:
                return await _process_one_batch(idx, b)

        # Execute all batches in the current event loop
        async def _run_all_batches():
            tasks = [
                _guarded_process(batch_idx, batch)
                for batch_idx, batch in enumerate(batches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # Ensure we have an event loop to run async work
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(_run_all_batches())
        finally:
            loop.close()

        for res, batch in zip(results, batches, strict=False):
            if isinstance(res, Exception):
                failed_embeddings.extend([item["id"] for item in batch])
            else:
                total_embedded += int(res or 0)

        # Update embedding status in database
        with transaction.atomic():
            # Mark metadata as embedded
            video_metadata.is_embedded = True
            video_metadata.save()

            # Mark segments as embedded (only successful ones)
            if segments.exists():
                successful_segment_ids = [
                    item["id"]
                    for item in embedding_items
                    if item["metadata"]["type"] == "segment"
                    and item["id"] not in failed_embeddings
                ]

                if successful_segment_ids:
                    segments.filter(segment_id__in=successful_segment_ids).update(
                        is_embedded=True
                    )

        result = {
            "total_items": len(embedding_items),
            "total_embedded": total_embedded,
            "failed_embeddings": len(failed_embeddings),
            "batches_processed": len(batches),
            "metadata_embedded": True,
            "summary_embedded": transcript.summary is not None
            and transcript.summary.strip(),
            "transcript_chunks_embedded": len(
                [
                    item
                    for item in embedding_items
                    if item["metadata"]["type"] == "transcript_chunk"
                ]
            ),
            "segments_embedded": len(
                [
                    item
                    for item in embedding_items
                    if item["metadata"]["type"] == "segment"
                ]
            )
            - len([item for item in failed_embeddings if item.startswith(video_id)]),
            "video_id": video_id,
        }

        logger.info(f"Embedding completed for video {video_id}: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in sync embedding process: {e}")
        raise


@shared_task(
    bind=True,
    name="video_processor.embed_video_content",
    soft_time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["embedding_soft_limit"],
    time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["embedding_hard_limit"],
    max_retries=YOUTUBE_CONFIG["RETRY_CONFIG"]["embedding"]["max_retries"],
    default_retry_delay=YOUTUBE_CONFIG["RETRY_CONFIG"]["embedding"]["countdown"],
    autoretry_for=(Exception,),
    retry_backoff=YOUTUBE_CONFIG["RETRY_CONFIG"]["embedding"]["backoff"],
    retry_jitter=YOUTUBE_CONFIG["RETRY_CONFIG"]["embedding"]["jitter"],
)
@idempotent_task
def embed_video_content(self, summary_result, url_request_id):
    """
    Embed video content using 4-layer strategy after summary generation.

    Args:
        summary_result (dict): Result from previous summary generation task
        url_request_id (str): UUID of the URLRequestTable to process

    Returns:
        dict: Embedding results with total items embedded and processing stats

    Raises:
        Exception: If embedding process fails after retries
    """
    url_request = None
    video_metadata = None
    transcript = None

    try:
        # Set initial progress
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(0, 100, "Generating embeddings")
        update_task_progress(
            self, TASK_STATES.get("EMBEDDING_CONTENT", "Embedding Content"), 10
        )

        # Get video metadata and transcript
        url_request = URLRequestTable.objects.select_related("video_metadata").get(
            request_id=url_request_id
        )

        # Check if video is excluded (skip embedding for excluded videos)
        if url_request.failure_reason == "excluded":
            video_id = getattr(url_request, "video_metadata", None)
            video_id = (
                getattr(video_id, "video_id", "unknown") if video_id else "unknown"
            )
            logger.info(
                f"Skipping embedding for excluded video {video_id} (reason: {url_request.failure_reason})"
            )
            return {
                "skipped": True,
                "reason": "excluded",
                "video_id": video_id,
                "total_items": 0,
                "total_embedded": 0,
                "already_embedded": False,
            }

        if not hasattr(url_request, "video_metadata"):
            raise ValueError("VideoMetadata not found for this request")

        video_metadata = url_request.video_metadata

        if not hasattr(video_metadata, "video_transcript"):
            raise ValueError("VideoTranscript not found for this request")

        transcript = video_metadata.video_transcript

        logger.info(f"Starting embedding process for video {video_metadata.video_id}")

        update_task_progress(
            self, TASK_STATES.get("EMBEDDING_CONTENT", "Embedding Content"), 30
        )

        # Check if already embedded (improved idempotency with vector store verification)
        if video_metadata.is_embedded:
            # Verify embeddings actually exist in vector store
            try:
                # Get shared vector service instance
                from ai_utils.config import get_config

                get_config()
                vector_service = get_vector_service()

                # Check if vectors exist for this video in the vector store
                test_query = VectorQuery(
                    query=f"video content from {video_metadata.title or video_metadata.video_id}",
                    top_k=1,
                    filters={"video_id": video_metadata.video_id},
                )

                # Create event loop for idempotency check
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    search_result = loop.run_until_complete(
                        vector_service.search_similar(query=test_query)
                    )
                finally:
                    loop.close()

                if search_result.results and len(search_result.results) > 0:
                    logger.info(
                        f"Video {video_metadata.video_id} already embedded and verified in vector store, skipping"
                    )
                    return {
                        "total_items": 0,
                        "total_embedded": 0,
                        "already_embedded": True,
                        "video_id": video_metadata.video_id,
                    }
                else:
                    logger.warning(
                        f"Video {video_metadata.video_id} marked as embedded but no vectors found in store, re-embedding"
                    )
                    # Reset the flag and continue with embedding
                    video_metadata.is_embedded = False
                    video_metadata.save()

            except Exception as e:
                logger.warning(
                    f"Could not verify vector store embeddings for {video_metadata.video_id}: {e}, re-embedding"
                )
                # Reset the flag and continue with embedding
                video_metadata.is_embedded = False
                video_metadata.save()

        # Perform embedding using synchronous function
        embedding_result = embed_video_content_sync(video_metadata, transcript)

        update_task_progress(
            self, TASK_STATES.get("EMBEDDING_CONTENT", "Embedding Content"), 90
        )

        # Log final results
        logger.info(
            f"Embedding completed for video {video_metadata.video_id}: "
            f"{embedding_result['total_embedded']}/{embedding_result['total_items']} items embedded"
        )

        update_task_progress(
            self, TASK_STATES.get("EMBEDDING_CONTENT", "Embedding Content"), 100
        )

        # Set final progress
        progress_recorder.set_progress(100, 100, "Embeddings complete")

        return embedding_result

    except SoftTimeLimitExceeded:
        # Task is approaching timeout - save status and exit gracefully
        logger.warning(
            f"Embedding generation soft timeout reached for request {url_request_id}"
        )

        try:
            # Mark embedding as failed due to timeout
            if video_metadata:
                video_metadata.is_embedded = False
                video_metadata.save()
                logger.error(
                    f"Marked embedding generation as failed due to timeout: {video_metadata.video_id}"
                )

        except Exception as cleanup_error:
            logger.error(
                f"Failed to update embedding status during timeout cleanup: {cleanup_error}"
            )

        # Re-raise with specific timeout message
        raise Exception(f"Embedding generation timeout for request {url_request_id}")

    except Exception as e:
        logger.error(f"Embedding failed for request {url_request_id}: {e}")

        # Mark as failed but don't stop the chain
        try:
            with transaction.atomic():
                if not url_request:
                    url_request = URLRequestTable.objects.select_related(
                        "video_metadata"
                    ).get(request_id=url_request_id)

                if hasattr(url_request, "video_metadata"):
                    video_metadata = url_request.video_metadata
                    video_metadata.is_embedded = False
                    video_metadata.save()

        except Exception as db_error:
            logger.error(f"Failed to update embedding status: {db_error}")

        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task(
                "embed_video_content", self.request.id, [url_request_id], {}, e
            )

        # Return error result but don't break the chain
        return {
            "total_items": 0,
            "total_embedded": 0,
            "failed_embeddings": 0,
            "video_id": "Unknown",
            "error": str(e),
        }
