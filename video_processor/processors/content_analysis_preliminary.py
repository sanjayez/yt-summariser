"""
Phase 1: Preliminary Content Analysis Task
Handles content analysis without timestamps - runs in parallel with summary/classification.
"""

import asyncio
from typing import Any

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction
from django.utils import timezone

from api.models import URLRequestTable
from telemetry import get_logger

from ..config import YOUTUBE_CONFIG
from ..models import ContentAnalysis

# Import the existing functions from the current content_analyzer.py
# We'll reuse these functions in our preliminary analysis
from .content_analyzer import (
    aggregate_results,
    chunk_transcript,
    process_chunks_parallel_async,
)

logger = get_logger(__name__)


@shared_task(
    bind=True,
    name="video_processor.content_analysis_preliminary",
    soft_time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"].get(
        "content_analysis_soft_limit", 600
    ),
    time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"].get("content_analysis_hard_limit", 900),
    max_retries=2,
    default_retry_delay=60,
)
def content_analysis_preliminary(self, transcript_result, url_request_id):
    """
    Phase 1: Celery task for preliminary content analysis.

    This task:
    1. Extracts transcript data (Django ORM - sync)
    2. Runs preliminary analysis without timestamps (async)
    3. Saves results to ContentAnalysis model (Django ORM - sync)

    Designed to run in parallel with summary generation and classification.

    Args:
        transcript_result: Result from transcript extraction task
        url_request_id: UUID of the URLRequestTable to process

    Returns:
        dict: Preliminary analysis results summary
    """
    content_analysis = None
    video_transcript = None

    try:
        logger.info(
            f"Starting preliminary content analysis for request {url_request_id}"
        )

        # Check if video was excluded in previous stage
        if transcript_result and transcript_result.get("excluded"):
            logger.info(
                f"Video was excluded in previous stage: {transcript_result.get('exclusion_reason')}"
            )
            return {
                "video_id": transcript_result.get("video_id"),
                "excluded": True,
                "exclusion_reason": transcript_result.get("exclusion_reason"),
                "skip_reason": "excluded_in_previous_stage",
            }

        # Django ORM operations (sync) - get transcript data
        url_request = URLRequestTable.objects.select_related(
            "video_metadata", "video_metadata__video_transcript"
        ).get(request_id=url_request_id)

        if not hasattr(url_request, "video_metadata") or not url_request.video_metadata:
            raise ValueError("VideoMetadata not found")

        video_metadata = url_request.video_metadata

        if (
            not hasattr(video_metadata, "video_transcript")
            or not video_metadata.video_transcript
        ):
            raise ValueError("VideoTranscript not found")

        video_transcript = video_metadata.video_transcript
        video_id = video_transcript.video_id

        if (
            not video_transcript.transcript_text
            or not video_transcript.transcript_text.strip()
        ):
            raise ValueError("No transcript text available")

        logger.info(
            f"Processing preliminary analysis for video {video_id}: {video_metadata.title}"
        )

        # Create or get ContentAnalysis record
        with transaction.atomic():
            content_analysis, created = ContentAnalysis.objects.get_or_create(
                video_transcript=video_transcript,
                defaults={"preliminary_analysis_status": "processing"},
            )

            if not created:
                content_analysis.preliminary_analysis_status = "processing"
                content_analysis.save(update_fields=["preliminary_analysis_status"])

        # Run async preliminary analysis (isolated from Django ORM)
        preliminary_results = asyncio.run(
            process_preliminary_analysis_async(
                transcript_text=video_transcript.transcript_text
            )
        )

        # Save preliminary results to ContentAnalysis (sync Django ORM)
        with transaction.atomic():
            content_analysis.raw_ad_segments = preliminary_results["raw_ad_segments"]
            content_analysis.raw_filler_segments = preliminary_results[
                "raw_filler_segments"
            ]
            content_analysis.speaker_tones = preliminary_results["speaker_tones"]
            content_analysis.preliminary_analysis_status = "completed"
            content_analysis.preliminary_completed_at = timezone.now()
            content_analysis.save()

        logger.info(f"Preliminary content analysis complete for {video_id}")

        return {
            "video_id": video_id,
            "preliminary_complete": True,
            "raw_ad_segments_count": len(preliminary_results["raw_ad_segments"]),
            "raw_filler_segments_count": len(
                preliminary_results["raw_filler_segments"]
            ),
            "speaker_tones": preliminary_results["speaker_tones"],
            "primary_tone": preliminary_results["primary_tone"],
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Preliminary analysis timeout for request {url_request_id}")

        if content_analysis:
            content_analysis.preliminary_analysis_status = "failed"
            content_analysis.save(update_fields=["preliminary_analysis_status"])

        raise

    except Exception as e:
        logger.error(f"Preliminary analysis failed for request {url_request_id}: {e}")

        if content_analysis:
            content_analysis.preliminary_analysis_status = "failed"
            content_analysis.save(update_fields=["preliminary_analysis_status"])

        # Return error result but don't break the parallel group
        return {
            "video_id": video_transcript.video_id if video_transcript else "unknown",
            "preliminary_complete": False,
            "error": str(e),
            "raw_ad_segments_count": 0,
            "raw_filler_segments_count": 0,
            "speaker_tones": ["neutral"],
        }


async def process_preliminary_analysis_async(transcript_text: str) -> dict[str, Any]:
    """
    Phase 1: Preliminary content analysis without timestamps.

    This function performs:
    1. Transcript chunking
    2. Parallel LLM analysis (ads/filler detection + tone analysis)
    3. Result aggregation and deduplication

    NO timestamps are added - that happens in Phase 2 after embedding.
    NO ratios are calculated - they require timestamps.

    Args:
        transcript_text: Full transcript text to analyze

    Returns:
        dict: Preliminary analysis results with raw segments and tones
    """
    try:
        logger.info("Starting preliminary content analysis (Phase 1)")

        # Step 1: Chunk the transcript (reuse existing function)
        chunks = chunk_transcript(transcript_text)
        logger.info(f"Created {len(chunks)} chunks for preliminary analysis")

        # Step 2: Process chunks in parallel with LLM (reuse existing function)
        chunk_results = await process_chunks_parallel_async(chunks)
        logger.info(f"Processed {len(chunk_results)} chunks in parallel")

        # Step 3: Aggregate and deduplicate results (reuse existing function)
        aggregated = aggregate_results(chunk_results)
        logger.info(
            f"Found {len(aggregated['ad_segments'])} ads, {len(aggregated['filler_segments'])} fillers"
        )
        logger.info(f"Detected tones: {aggregated['speaker_tones']}")

        # Return preliminary results - NO timestamps, NO ratios
        return {
            "raw_ad_segments": aggregated["ad_segments"],  # Text excerpts only
            "raw_filler_segments": aggregated["filler_segments"],  # Text excerpts only
            "speaker_tones": aggregated["speaker_tones"],  # Tone analysis
            "primary_tone": aggregated["primary_tone"],
            "tone_evidence": aggregated.get("tone_evidence", []),
        }

    except Exception as e:
        logger.error(f"Preliminary analysis failed: {e}")
        raise
