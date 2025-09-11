"""Database operations for transcript processing"""

from django.db import transaction

from yt_workflow.models import TranscriptSegment
from yt_workflow.transcript.types import NormalizedLine


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

    # Use bulk_create for efficient batch insertion
    with transaction.atomic():
        created_segments = TranscriptSegment.objects.bulk_create(
            segments_to_create,
            batch_size=100,  # Process in batches of 100
            ignore_conflicts=False,  # Raise error on duplicate line_id
        )

    return len(created_segments)
