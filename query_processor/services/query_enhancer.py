"""
LLM Query Enhancement Service
Clean, focused service for enhancing search queries using LLM
"""

import json

from ai_utils.services.registry import get_gemini_llm_service
from telemetry.logging.logger import get_logger

from ..prompts import get_enhancement_prompt

logger = get_logger(__name__)


class QueryEnhancementService:
    """Service for enhancing search queries using LLM."""

    def __init__(self):
        """Initialize with LLM service."""
        self.llm_service = get_gemini_llm_service()

    async def enhance_query(
        self, original_query: str, model: str | None = None
    ) -> tuple[list[str], list[str], str]:
        try:
            # Direct LLM call for query enhancement
            prompt = get_enhancement_prompt(original_query)

            # Call LLM service for query enhancement
            llm_response = await self.llm_service.generate_text(
                prompt, temperature=0.3, model=model
            )

            response_text = llm_response.get("text", "")

            # Parse LLM response
            try:
                result = json.loads(response_text)
                concepts = result.get("concepts", [original_query])
                enhanced_queries = result.get("enhanced_queries", [original_query])
                intent_type = result.get("intent_type", "LOOKUP")

                logger.info(
                    f"Query enhancement completed: {len(enhanced_queries)} queries generated, intent classified as {intent_type}"
                )
                return concepts, enhanced_queries, intent_type

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response: {e}")
                return [original_query], [original_query], "LOOKUP"

        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return [original_query], [original_query], "LOOKUP"
