"""Exact search utilities for chapter timestamp mapping"""

import bisect
import logging

from yt_workflow.transcript.models import TranscriptSegment

logger = logging.getLogger(__name__)


def normalize_text_aggressive(text: str) -> str:
    """
    Args: text: Input text to normalize

    Returns:
        Normalized string (lowercase, alphanumeric only)
    """
    return "".join(c.lower() for c in text if c.isalnum())


def build_transcript_mapping(video_id: str) -> tuple[str, list[dict]]:
    """
    Build concatenated transcript with character position mapping

    Args: video_id: Video identifier

    Returns: Tuple of (normalized_text, char_position_map)

    Raises:
        ValueError: If no segments found for video_id
    """
    segments = list(
        TranscriptSegment.objects.filter(video_id=video_id).order_by(
            "idx"
        )  # CRITICAL: Sort by idx for binary search
    )

    if not segments:
        raise ValueError(f"No transcript segments found for video {video_id}")

    normalized_text = ""
    char_position_map = []
    current_char_pos = 0

    for segment in segments:
        original_text = segment.text
        normalized_segment = normalize_text_aggressive(original_text)

        if normalized_segment:  # Skip empty normalized segments
            char_position_map.append(
                {
                    "start_char": current_char_pos,
                    "end_char": current_char_pos + len(normalized_segment),
                    "segment": segment,
                }
            )

            normalized_text += normalized_segment
            current_char_pos += len(normalized_segment)

    return normalized_text, char_position_map


def find_text_position(normalized_query: str, normalized_text: str) -> int:
    """
    Find exact position of query in text using str.find()

    Args:
        normalized_query: Normalized query string
        normalized_text: Normalized concatenated transcript

    Returns:
        Character position (0-based) or -1 if not found
    """
    return normalized_text.find(normalized_query)


def map_position_to_timestamp(
    char_pos: int, char_position_map: list[dict]
) -> float | None:
    """
    Map character position to segment start timestamp using binary search

    Args:
        char_pos: Character position in concatenated text
        char_position_map: List of character position mappings (sorted by start_char)

    Returns:
        Start timestamp (float) or None if position not found
    """
    if not char_position_map or char_pos < 0:
        return None

    # Extract start_char values for binary search
    start_chars = [mapping["start_char"] for mapping in char_position_map]

    # Find the rightmost position where start_char <= char_pos
    idx = bisect.bisect_right(start_chars, char_pos) - 1

    # Check if we found a valid mapping
    if idx >= 0 and idx < len(char_position_map):
        mapping = char_position_map[idx]

        # Verify char_pos falls within this mapping's range
        if mapping["start_char"] <= char_pos < mapping["end_char"]:
            segment = mapping["segment"]
            return float(segment.start)

    return None


def find_anchors(chapters: list[dict], video_id: str) -> list[dict]:
    """
    Main function to find timestamps for chapter start_strings using exact search

    Args:
        chapters: List of chapter dicts with start_string field
        video_id: Video identifier

    Returns:
        Updated chapters list with timestamp field added

    Raises:
        ValueError: If no segments found for video_id
    """
    if not chapters:
        return chapters

    try:
        # Build transcript mapping once for all chapters
        normalized_text, char_position_map = build_transcript_mapping(video_id)

        # Process each chapter
        updated_chapters = []

        for chapter in chapters:
            # Create a copy to avoid modifying original
            updated_chapter = chapter.copy()

            start_string = chapter.get("start_string", "")

            if not start_string:
                # No start_string to search for
                updated_chapter["timestamp"] = None
                updated_chapters.append(updated_chapter)
                continue

            # Normalize query and search
            normalized_query = normalize_text_aggressive(start_string)

            if not normalized_query:
                # Query normalized to empty string
                updated_chapter["timestamp"] = None
                updated_chapters.append(updated_chapter)
                continue

            # Find exact position
            char_pos = find_text_position(normalized_query, normalized_text)

            if char_pos == -1:
                # Exact match not found
                updated_chapter["timestamp"] = None
                updated_chapters.append(updated_chapter)
                continue

            # Map to timestamp
            start_time = map_position_to_timestamp(char_pos, char_position_map)

            updated_chapter["timestamp"] = start_time

            updated_chapters.append(updated_chapter)

        return updated_chapters

    except Exception as e:
        # Log error and return chapters with null timestamps
        logger.error(f"Exact search failed for video {video_id}: {str(e)}")

        # Return chapters with null timestamps
        return [{**chapter, "timestamp": None} for chapter in chapters]
