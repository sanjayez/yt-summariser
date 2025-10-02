"""Zero-shot chapter detection using single LLM call with verbatim boundaries"""

import re
from typing import Any

from llama_index.core.program import LLMTextCompletionProgram

from ai_utils.services.registry import get_gemini_llm_service
from yt_workflow.transcript.prompts.chapter_detect_prompt import CHAPTER_PROMPT
from yt_workflow.transcript.types import ChapterDetectionOutput, MacroChunk


def normalize_text(text: str) -> str:
    """Normalize text: whitespace normalization only, preserving case and punctuation for verbatim boundary matching"""
    # Replace multiple whitespaces with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


async def detect_chapters(macros: list[MacroChunk], video_id: str) -> dict[str, Any]:
    """
    Zero-shot chapter detection using verbatim boundary matching.

    Args:
        macros: List of MacroChunk objects
        video_id: Video identifier

    Returns:
        Dictionary containing detected chapters with verbatim boundaries
    """

    # Concatenate and normalize macro text
    full_text = " ".join(m.text for m in macros)
    normalized_text = normalize_text(full_text)

    # Build prompt
    prompt = f"{CHAPTER_PROMPT}\n\nTRANSCRIPT:\n{normalized_text}"

    # Get LLM service and access underlying provider
    llm_service = get_gemini_llm_service()

    try:
        # Get the LLM client with specific parameters
        llm_client = llm_service.provider.get_llm_client(
            model="gemini-2.5-flash-lite",
            temperature=0.1,  # Low temperature for consistent verbatim matching
            max_tokens=8000,
        )

        # Create structured output program
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ChapterDetectionOutput,
            prompt_template_str="{transcript_text}",
            llm=llm_client,
        )

        # Execute program with full prompt
        result = await program.acall(transcript_text=prompt)

        # Convert to dict format
        return {
            "chapters": [chapter.model_dump() for chapter in result.chapters],
            "method": result.method,
        }

    except Exception as e:
        return {"chapters": [], "error": str(e)}
