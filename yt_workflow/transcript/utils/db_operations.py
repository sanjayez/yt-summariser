"""Database operations for transcript processing"""

from django.db import transaction

from telemetry import get_logger
from yt_workflow.models import TranscriptSegment, VideoTable
from yt_workflow.transcript.types import NormalizedLine

logger = get_logger(__name__)


def batch_insert_transcripts(lines: list[NormalizedLine]) -> int:
    """
    Batch insert normalized lines into TranscriptSegment table.

    Args:
        lines: List of NormalizedLine objects to insert

    Returns:
        int: Number of segments created
    """
    if not lines:
        return 0

    # Convert NormalizedLine objects to TranscriptSegment objects
    segments_to_create = [
        TranscriptSegment(
            line_id=line.line_id,
            video_id=line.line_id.split("_line_")[0],  # Extract video_id from line_id
            idx=line.idx,
            start=line.start,
            end=line.end,
            text=line.text,
        )
        for line in lines
    ]

    # Calculate dynamic batch size
    batch_size = min(500, len(segments_to_create))

    # Use bulk_create for efficient batch insertion
    with transaction.atomic():
        created_segments = TranscriptSegment.objects.bulk_create(
            segments_to_create,
            batch_size=batch_size,  # Dynamic batch size with 500 minimum
            ignore_conflicts=False,  # Raise error on duplicate line_id
        )

    return len(created_segments)


def save_chapters_to_video_table(
    timestamped_chapters: list[dict], video_id: str
) -> None:
    """
    Save timestamped chapters to VideoTable with atomic transaction and logging.

    Args:
        timestamped_chapters: List of chapter dicts with timestamp field
        video_id: Video identifier
    """
    with transaction.atomic():
        video_record, created = VideoTable.objects.get_or_create(video_id=video_id)
        if not video_record.chapters:
            video_record.chapters = timestamped_chapters
            video_record.save()

    successful_count = len(
        [ch for ch in timestamped_chapters if ch.get("timestamp") is not None]
    )
    logger.info(
        f"Saved {successful_count}/{len(timestamped_chapters)} chapters for {video_id}"
    )


def merge_chapter_summaries(chapters: list[dict], summaries: list[dict]) -> list[dict]:
    """
    Merge bullet points from summaries into chapter objects by matching titles.

    Args:
        chapters: Original chapters with timestamps
        summaries: Chapter summaries with bullet_points

    Returns:
        Enhanced chapters with bullet_points added
    """
    if not chapters or not summaries:
        return chapters

    # Create a mapping of summary title to bullet_points for fast lookup
    summary_map = {summary["title"]: summary["bullet_points"] for summary in summaries}

    # Add bullet_points to matching chapters
    enhanced_chapters = []
    for chapter in chapters:
        enhanced_chapter = chapter.copy()
        chapter_title = chapter.get("chapter", "")

        # Match by title and add bullet_points if found
        if chapter_title in summary_map:
            enhanced_chapter["bullet_points"] = summary_map[chapter_title]

        enhanced_chapters.append(enhanced_chapter)

    return enhanced_chapters


def update_chapters_with_summary(
    chapter_summaries: list[dict], executive_summary: dict, video_id: str
) -> None:
    """
    Update existing chapters with summaries and save executive summary.

    Args:
        chapter_summaries: List of chapter summaries with bullet_points
        executive_summary: Dict with executive_summary and key_highlights
        video_id: Video identifier
    """
    with transaction.atomic():
        video_record = VideoTable.objects.get(video_id=video_id)  # Must exist

        # Merge summaries into existing chapters
        if video_record.chapters and chapter_summaries:
            enhanced_chapters = merge_chapter_summaries(
                video_record.chapters, chapter_summaries
            )
            video_record.chapters = enhanced_chapters

        # Save executive summary
        if executive_summary:
            video_record.executive_summary = executive_summary

        video_record.save()

    logger.info(
        f"Updated {len(chapter_summaries)} chapters with summaries and executive summary for {video_id}"
    )
