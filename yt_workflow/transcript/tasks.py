import asyncio

from celery import shared_task

from telemetry import get_logger
from yt_workflow.shared.clients.broker_client import get_transcript
from yt_workflow.shared.vector_operations import upsert_transcript_chunks
from yt_workflow.transcript.utils import (
    _normalize_lines,
    assign_primary_macros,
    batch_insert_transcripts,
    build_macro_chunks,
    build_micro_chunks,
    detect_chapters,
    find_anchors,
)

logger = get_logger(__name__)


@shared_task(bind=True, name="yt_workflow.process_transcript")
def process_transcript(self, video_id: str) -> None:  # type: ignore
    """Process transcript for a single video"""
    try:
        # Fetch transcript
        transcript_response = get_transcript(video_id)
        segments = transcript_response["data"]["segments"]

        # Normalize segments into lines
        lines = _normalize_lines(segments, video_id)

        # Batch insert transcript segments into database
        batch_insert_transcripts(lines)

        # Build micro chunks with overlap
        micro_chunks = build_micro_chunks(lines, video_id)

        # Build macro chunks without overlap
        macro_chunks = build_macro_chunks(lines, video_id)

        # Assign micro chunks to their primary macros
        assign_primary_macros(micro_chunks, macro_chunks)

        # Upsert chunks to vector store and run chapter detection in parallel
        async def process_transcript_chunks():
            return await asyncio.gather(
                upsert_transcript_chunks(micro_chunks, macro_chunks, video_id),
                detect_chapters(macro_chunks, video_id),
            )

        vector_result, chapters_result = asyncio.run(process_transcript_chunks())

        # Map chapter boundaries to timestamps using exact search
        timestamped_chapters = []
        if chapters_result and "chapters" in chapters_result:
            try:
                # Use exact search for timestamp mapping
                timestamped_chapters = find_anchors(
                    chapters_result["chapters"], video_id
                )

                # Log results
                successful_count = len(
                    [
                        ch
                        for ch in timestamped_chapters
                        if ch.get("timestamp") is not None
                    ]
                )
                total_count = len(timestamped_chapters)
                logger.info(
                    f"Exact search results for {video_id}: {successful_count}/{total_count} chapters found timestamps"
                )

            except Exception as e:
                logger.error(f"Exact search failed for {video_id}: {str(e)}")
                # Fallback: chapters without timestamps
                timestamped_chapters = [
                    {**ch, "timestamp": None} for ch in chapters_result["chapters"]
                ]

        # TODO
        # to save timestamped_chapters to database
        # to implement retry mechanism for failed vector upserts

    except Exception as e:
        # TODO
        # Should mark YT Insight Run (New version of URLRequest Table) as failed with the error message
        # After 3 retries
        logger.error(f"Process transcript failed for {video_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error("Exception traceback:", exc_info=True)
        raise
