"""
Text chunking and embedding preparation utilities using LlamaIndex.
These utilities are reusable across the video processing pipeline.
"""

import logging
from typing import Any

from llama_index.core.schema import Document
from llama_index.core.text_splitter import SentenceSplitter

logger = logging.getLogger(__name__)


def chunk_transcript_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Chunk full transcript text for embedding using LlamaIndex.

    Args:
        text: Full transcript text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks ready for embedding
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Use LlamaIndex's SentenceSplitter for intelligent chunking
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]\\s+",
        )

        # Create document and split
        document = Document(text=text)
        chunks = splitter.split_text(document.text)

        logger.info(f"Successfully chunked text into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking transcript text: {e}")
        # Fallback to simple splitting if LlamaIndex fails
        words = text.split()
        chunk_word_count = chunk_size // 5  # Rough estimate: 5 chars per word
        chunks = []

        for i in range(0, len(words), chunk_word_count):
            chunk = " ".join(words[i : i + chunk_word_count])
            chunks.append(chunk)

        logger.warning(f"Used fallback chunking, created {len(chunks)} chunks")
        return chunks


def format_metadata_for_embedding(metadata_obj) -> str:
    """
    Format VideoMetadata object into searchable text for embedding.

    Args:
        metadata_obj: VideoMetadata instance

    Returns:
        Formatted text string ready for embedding
    """
    try:
        # Build comprehensive metadata text
        parts = []

        # Basic info
        if metadata_obj.title:
            parts.append(f"Title: {metadata_obj.title}")

        if metadata_obj.description:
            # Truncate very long descriptions
            desc = (
                metadata_obj.description[:500] + "..."
                if len(metadata_obj.description) > 500
                else metadata_obj.description
            )
            parts.append(f"Description: {desc}")

        if metadata_obj.channel_name:
            parts.append(f"Channel: {metadata_obj.channel_name}")

        # Additional context
        if metadata_obj.duration:
            parts.append(f"Duration: {metadata_obj.duration_string}")

        if metadata_obj.view_count:
            parts.append(f"Views: {metadata_obj.view_count:,}")

        if metadata_obj.like_count:
            parts.append(f"Likes: {metadata_obj.like_count:,}")

        if metadata_obj.upload_date:
            parts.append(f"Published: {metadata_obj.upload_date}")

        if metadata_obj.language:
            parts.append(f"Language: {metadata_obj.language}")

        # Tags and categories
        if metadata_obj.tags:
            tags_text = ", ".join(metadata_obj.tags[:10])  # Limit to first 10 tags
            parts.append(f"Tags: {tags_text}")

        if metadata_obj.categories:
            categories_text = ", ".join(metadata_obj.categories)
            parts.append(f"Categories: {categories_text}")

        # Channel info
        if metadata_obj.channel_follower_count:
            parts.append(
                f"Channel Subscribers: {metadata_obj.channel_follower_count:,}"
            )

        if metadata_obj.channel_is_verified:
            parts.append("Channel: Verified")

        if metadata_obj.uploader_id:
            parts.append(f"Channel Handle: {metadata_obj.uploader_id}")

        formatted_text = "\n".join(parts)

        logger.debug(
            f"Formatted metadata for video {metadata_obj.video_id}: {len(formatted_text)} characters"
        )
        return formatted_text

    except Exception as e:
        logger.error(f"Error formatting metadata for embedding: {e}")
        # Fallback to basic info
        return f"Title: {getattr(metadata_obj, 'title', 'Unknown')}\nChannel: {getattr(metadata_obj, 'channel_name', 'Unknown')}"


def format_summary_for_embedding(summary: str, key_points: list[str]) -> str:
    """
    Format summary and key points for embedding.

    Args:
        summary: Generated summary text
        key_points: List of key points

    Returns:
        Formatted text combining summary and key points
    """
    try:
        parts = []

        if summary and summary.strip():
            parts.append(f"Summary: {summary}")

        if key_points and isinstance(key_points, list) and len(key_points) > 0:
            # Filter out empty key points
            valid_points = [point for point in key_points if point and point.strip()]
            if valid_points:
                points_text = "\n".join([f"• {point}" for point in valid_points])
                parts.append(f"Key Points:\n{points_text}")

        formatted_text = "\n\n".join(parts)

        logger.debug(
            f"Formatted summary for embedding: {len(formatted_text)} characters"
        )
        return formatted_text

    except Exception as e:
        logger.error(f"Error formatting summary for embedding: {e}")
        return summary or "No summary available"


def format_segment_for_embedding(segment_obj) -> str:
    """
    Format TranscriptSegment for embedding with timestamp context.

    Args:
        segment_obj: TranscriptSegment instance

    Returns:
        Formatted segment text with timestamp for embedding
    """
    try:
        # Format: [MM:SS] segment_text
        timestamp = segment_obj.get_formatted_timestamp()
        text = segment_obj.text.strip()

        # Add some context about the video if available
        video_context = ""
        if hasattr(segment_obj, "transcript") and hasattr(
            segment_obj.transcript, "video_metadata"
        ):
            video_title = segment_obj.transcript.video_metadata.title
            if video_title:
                video_context = (
                    f" (from: {video_title[:50]}...)"
                    if len(video_title) > 50
                    else f" (from: {video_title})"
                )

        formatted_text = f"[{timestamp}] {text}{video_context}"

        logger.debug(
            f"Formatted segment {segment_obj.segment_id} for embedding: {len(formatted_text)} characters"
        )
        return formatted_text

    except Exception as e:
        logger.error(f"Error formatting segment for embedding: {e}")
        return getattr(segment_obj, "text", "No text available")


def validate_embedding_text(text: str, max_length: int = 8000) -> str:
    """
    Validate and truncate text for embedding if necessary.

    Args:
        text: Text to validate
        max_length: Maximum allowed length

    Returns:
        Validated and potentially truncated text
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding validation")
            return "No content available"

        # Strip whitespace
        text = text.strip()

        # Truncate if too long
        if len(text) > max_length:
            logger.warning(
                f"Text too long ({len(text)} chars), truncating to {max_length}"
            )
            text = text[: max_length - 3] + "..."

        return text

    except Exception as e:
        logger.error(f"Error validating embedding text: {e}")
        return "Error processing text"


def prepare_batch_embeddings(
    items: list[dict[str, Any]], batch_size: int = 15
) -> list[list[dict[str, Any]]]:
    """
    Prepare embedding items for batch processing.

    Args:
        items: List of embedding items with 'id' and 'text' keys
        batch_size: Number of items per batch

    Returns:
        List of batches, each containing up to batch_size items
    """
    try:
        if not items:
            logger.warning("No items provided for batch preparation")
            return []

        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batches.append(batch)

        logger.info(f"Prepared {len(batches)} batches from {len(items)} items")
        return batches

    except Exception as e:
        logger.error(f"Error preparing batch embeddings: {e}")
        return [[item] for item in items]  # Fallback to individual items
