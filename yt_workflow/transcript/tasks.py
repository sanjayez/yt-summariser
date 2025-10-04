import asyncio

from celery import shared_task

from telemetry import get_logger
from yt_workflow.shared.clients.broker_client import get_transcript
from yt_workflow.shared.models import VideoTable
from yt_workflow.shared.vector_operations import upsert_transcript_chunks
from yt_workflow.transcript.models import TranscriptSegment
from yt_workflow.transcript.utils import (
    _normalize_lines,
    assign_primary_macros,
    batch_insert_transcripts,
    build_chapter_chunks,
    build_macro_chunks,
    build_micro_chunks,
    detect_chapters,
    extract_chapter_ranges,
    find_anchors,
    generate_executive_summary,
    save_chapters_to_video_table,
    summarize_chapter_chunks,
    update_chapters_with_summary,
)

logger = get_logger(__name__)


@shared_task(bind=True, name="yt_workflow.process_transcript")
def process_transcript(self, video_id: str) -> None:  # type: ignore
    """Process transcript for a single video"""
    try:
        # Early check - skip if both segments and chapters exist
        segments_exist = TranscriptSegment.objects.filter(video_id=video_id).exists()
        video_record, created = VideoTable.objects.get_or_create(video_id=video_id)

        if segments_exist and video_record.chapters:
            logger.info(f"Video {video_id} already fully processed, skipping")
            return

        # Fetch transcript
        transcript_response = get_transcript(video_id)
        segments = transcript_response["data"]["segments"]

        # Normalize segments into lines
        lines = _normalize_lines(segments, video_id)

        # Batch insert transcript segments into database
        batch_insert_transcripts(lines)

        # Need to re-think the role and construction of vector chunks

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
                timestamped_chapters = find_anchors(
                    chapters_result["chapters"], video_id
                )

                # Save chapters to VideoTable
                save_chapters_to_video_table(timestamped_chapters, video_id)

                # Extract chapter ranges using final transcript timestamp
                final_timestamp = lines[-1].end
                chapter_ranges = extract_chapter_ranges(
                    timestamped_chapters, final_timestamp
                )

                # Build chapter chunks using database queries
                chapter_chunks = build_chapter_chunks(chapter_ranges, video_id)

                # Summarize chapters in parallel
                async def process_summaries():
                    return await summarize_chapter_chunks(chapter_chunks, video_id)

                chapter_summaries = asyncio.run(process_summaries())
                # logger.info(f"Generated {len(chapter_summaries)} chapter summaries for {video_id}")
                logger.info(f"chapter summaries for {video_id}: {chapter_summaries}")

                # Generate executive summary
                async def process_executive_summary():
                    return await generate_executive_summary(chapter_summaries, video_id)

                executive_summary = asyncio.run(process_executive_summary())
                logger.info(
                    f"Generated executive summary for {video_id}: {executive_summary}"
                )

                # Update chapters with summaries and save executive summary
                update_chapters_with_summary(
                    chapter_summaries, executive_summary, video_id
                )

            except Exception as e:
                logger.error(f"Chapter processing failed for {video_id}: {str(e)}")
                timestamped_chapters = [
                    {**ch, "timestamp": None} for ch in chapters_result["chapters"]
                ]

        # TODO
        # to implement retry mechanism for failed vector upserts

    except Exception as e:
        # TODO
        # Should mark YT Insight Run (New version of URLRequest Table) as failed with the error message
        # After 3 retries
        logger.error(f"Process transcript failed for {video_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error("Exception traceback:", exc_info=True)
        raise
