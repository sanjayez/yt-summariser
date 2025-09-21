"""Zero-shot chapter detection using single LLM call with verbatim boundaries"""

import json
import re
from typing import Any

from ai_utils.services.registry import get_gemini_llm_service
from yt_workflow.transcript.prompts.chapter_detect_prompt import CHAPTER_PROMPT
from yt_workflow.transcript.types.models import MacroChunk


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, trim whitespace, strip punctuation"""
    # Convert to lowercase
    text = text.lower()
    # Replace multiple whitespaces with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks"""
    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


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

    # Make LLM call
    llm_service = get_gemini_llm_service()

    response = await llm_service.generate_text(
        prompt=prompt,
        temperature=0.1,  # Low temperature for consistent verbatim matching
        max_tokens=4000,
        model="gemini-2.5-flash-lite",
    )

    # Parse response
    if response.get("status") == "failed":
        return {"chapters": [], "error": response.get("error")}

    try:
        clean_json = extract_json_from_response(response["text"])
        result = json.loads(clean_json)
        chapters = result.get("chapters", [])

        return {"chapters": chapters, "method": "verbatim_boundaries"}

    except json.JSONDecodeError as e:
        return {"chapters": [], "error": str(e)}
