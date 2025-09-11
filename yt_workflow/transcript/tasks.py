import logging

from celery import shared_task

from yt_workflow.shared.clients.broker_client import get_transcript
from yt_workflow.transcript.utils import (
    _normalize_lines,
    assign_primary_macros,
    build_macro_chunks,
    build_micro_chunks,
)

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="yt_workflow.process_transcript")
def process_transcript(self, video_id: str) -> None:  # type: ignore
    """Process transcript for a single video"""
    try:
        logger.info(f"Fetching transcript for video {video_id}")
        transcript_response = get_transcript(video_id)
        logger.info(f"Successfully fetched transcript for video {video_id}")

        segments = transcript_response["data"]["segments"]

        # Normalize segments into lines
        lines = _normalize_lines(segments, video_id)

        # Build micro chunks with overlap
        micro_chunks = build_micro_chunks(lines, video_id)

        # Build macro chunks without overlap
        macro_chunks = build_macro_chunks(lines, video_id)

        # Assign micro chunks to their primary macros
        assign_primary_macros(micro_chunks, macro_chunks)

        # TODO
        # to upsert micros and macros to weaviate
        # to analyse micros for bullet points / promotions / fillers (based on video title, theme of the current chunk)
        # to analyse macros for chapters

    except Exception as e:
        # TODO
        # Should mark YT Insight Run (New version of URLRequest Table) as failed with the error message
        # After 3 retries
        logger.error(f"Failed to fetch transcript for video {video_id}: {str(e)}")
