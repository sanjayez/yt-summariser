"""Chapter processing utilities for transcript workflow"""

import asyncio
from typing import cast

from llama_index.core.program import LLMTextCompletionProgram

from ai_utils.services.registry import get_gemini_llm_service
from telemetry import get_logger
from yt_workflow.transcript.models import TranscriptSegment
from yt_workflow.transcript.prompts.chapter_summary_prompt import (
    CHAPTER_SUMMARY_PROMPT,
)
from yt_workflow.transcript.prompts.executive_summary_prompt import (
    EXECUTIVE_SUMMARY_PROMPT,
)
from yt_workflow.transcript.types import ChapterSummary, ExecutiveSummary

logger = get_logger(__name__)


def extract_chapter_ranges(chapters: list[dict], final_timestamp: float) -> list[dict]:
    """
    Extract simple time ranges from chapters.

    Args:
        chapters: Already filtered and sorted chapters with timestamps
        final_timestamp: End timestamp from last transcript segment

    Returns:
        List of {"start": float, "end": float, "title": str}

    Example:
        >>> chapters = [
        ...     {"chapter": "Introduction", "timestamp": 0.0},
        ...     {"chapter": "Main Content", "timestamp": 180.5}
        ... ]
        >>> extract_chapter_ranges(chapters, 720.8)
        [
            {"start": 0.0, "end": 180.5, "title": "Introduction"},
            {"start": 180.5, "end": 720.8, "title": "Main Content"}
        ]
    """
    if not chapters:
        logger.debug("No chapters provided, returning empty ranges")
        return []

    ranges = []
    for i, chapter in enumerate(chapters):
        start = chapter["timestamp"]

        # End is next chapter's start, or final_timestamp for last chapter
        end = chapters[i + 1]["timestamp"] if i < len(chapters) - 1 else final_timestamp

        ranges.append(
            {
                "start": start,
                "end": end,
                "title": chapter.get("chapter", f"Chapter {i + 1}"),
            }
        )

    logger.info(f"Extracted {len(ranges)} chapter ranges from {len(chapters)} chapters")
    return ranges


def build_chapter_chunks(chapter_ranges: list[dict], video_id: str) -> list[dict]:
    """
    Build concatenated text chunks for each chapter using database queries.

    Args:
        chapter_ranges: Time ranges for each chapter
        video_id: Video identifier

    Returns:
        List of {"title": str, "start": float, "end": float, "text": str}
    """
    chapter_chunks = []

    for chapter_range in chapter_ranges:
        start = chapter_range["start"]
        end = chapter_range["end"]
        title = chapter_range["title"]

        # Database query with time range filter
        segments = (
            TranscriptSegment.objects.filter(
                video_id=video_id,
                start__lt=end,  # Segment starts before chapter ends
                end__gt=start,  # Segment ends after chapter starts
            )
            .order_by("start")
            .values_list("text", flat=True)
        )

        # Concatenate text
        text = " ".join(segments)

        chapter_chunks.append(
            {"title": title, "start": start, "end": end, "text": text}
        )

    logger.info(f"Created {len(chapter_chunks)} chapter chunks for {video_id}")
    return chapter_chunks


def create_empty_summary(chapter_chunk: dict) -> dict:
    """Create empty summary on failure"""
    return {
        "title": chapter_chunk["title"],
        "bullet_points": [],
        "start": chapter_chunk["start"],
        "end": chapter_chunk["end"],
    }


async def summarize_single_chapter(chapter_chunk: dict) -> dict:
    """
    Summarize a single chapter chunk using Gemini.

    Args:
        chapter_chunk: Dict with title, text, start, end

    Returns:
        Dict with title, bullet_points, start, end
    """
    try:
        # Build prompt with chapter content
        prompt = f"{CHAPTER_SUMMARY_PROMPT}\n\nCHAPTER: {chapter_chunk['title']}\nCONTENT:\n{chapter_chunk['text']}"

        # Get Gemini LLM service (same as detect_chapters)
        llm_service = get_gemini_llm_service()
        llm_client = llm_service.provider.get_llm_client(
            model="gemini-2.5-flash-lite",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=1000,
        )

        # Create structured output program
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ChapterSummary,
            prompt_template_str="{chapter_content}",
            llm=llm_client,
        )

        # Execute program
        result = await program.acall(chapter_content=prompt)

        # Return with timestamps from input data (not LLM)
        return {
            "title": result.title,
            "bullet_points": result.bullet_points,
            "start": chapter_chunk["start"],
            "end": chapter_chunk["end"],
        }

    except Exception as e:
        logger.error(
            f"Failed to summarize chapter '{chapter_chunk['title']}': {str(e)}"
        )
        return create_empty_summary(chapter_chunk)


async def summarize_chapter_chunks(
    chapter_chunks: list[dict], video_id: str
) -> list[dict]:
    """
    Parallel summarization of all chapter chunks.

    Args:
        chapter_chunks: List of chapter chunks with text
        video_id: Video identifier

    Returns:
        List of chapter summaries with bullet points
    """
    if not chapter_chunks:
        logger.debug(f"No chapter chunks to summarize for {video_id}")
        return []

    # Create parallel tasks for each chapter
    tasks = [summarize_single_chapter(chunk) for chunk in chapter_chunks]

    # Execute all tasks in parallel using asyncio.gather
    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle exceptions
    processed_summaries: list[dict] = []
    for i, summary in enumerate(summaries):
        if isinstance(summary, Exception):
            logger.error(f"Chapter {i} summarization failed: {str(summary)}")
            processed_summaries.append(create_empty_summary(chapter_chunks[i]))
        else:
            # MyPy type narrowing: summary is dict here, not BaseException
            processed_summaries.append(cast(dict, summary))

    logger.info(
        f"Completed summarization of {len(processed_summaries)} chapters for {video_id}"
    )
    return processed_summaries


def format_chapters_for_summary(chapter_summaries: list[dict]) -> str:
    """
    Convert chapter summaries to LLM-friendly text format.

    Args:
        chapter_summaries: List of chapter summaries with titles and bullet_points

    Returns:
        Formatted text string with chapters and bullets
    """
    formatted_sections = []

    for chapter in chapter_summaries:
        section = f"Chapter: {chapter['title']}\n"
        for bullet in chapter["bullet_points"]:
            section += f"• {bullet}\n"
        formatted_sections.append(section)

    return "\n".join(formatted_sections)


async def generate_executive_summary(
    chapter_summaries: list[dict], video_id: str
) -> dict:
    """
    Generate executive summary from chapter summaries using Gemini.

    Args:
        chapter_summaries: List of chapter summaries with titles and bullet_points
        video_id: Video identifier

    Returns:
        Dict with executive_summary and key_highlights
    """
    try:
        if not chapter_summaries:
            logger.debug(f"No chapter summaries to process for {video_id}")
            return {"executive_summary": "", "key_highlights": []}

        # Format chapters for LLM processing
        formatted_content = format_chapters_for_summary(chapter_summaries)

        # Build prompt with formatted chapters
        prompt = f"{EXECUTIVE_SUMMARY_PROMPT}\n\nCHAPTERS:\n{formatted_content}"

        # Get Gemini LLM service (same as other functions)
        llm_service = get_gemini_llm_service()
        llm_client = llm_service.provider.get_llm_client(
            model="gemini-2.5-flash-lite",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=800,
        )

        # Create structured output program
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ExecutiveSummary,
            prompt_template_str="{executive_content}",
            llm=llm_client,
        )

        # Execute program
        result = await program.acall(executive_content=prompt)

        # Return as dict
        return {
            "executive_summary": result.executive_summary,
            "key_highlights": result.key_highlights,
        }

    except Exception as e:
        logger.error(f"Failed to generate executive summary for {video_id}: {str(e)}")
        return {"executive_summary": "", "key_highlights": []}
