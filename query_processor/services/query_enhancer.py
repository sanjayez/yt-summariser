"""
LLM Query Enhancement Service
Clean, focused service for enhancing search queries using LLM
"""

import json

from ai_utils.services.registry import get_gemini_llm_service
from telemetry.logging.logger import get_logger
from telemetry.timing import timed_operation

from ..prompts import get_enhancement_prompt

logger = get_logger(__name__)


class QueryEnhancementService:
    """Service for enhancing search queries using LLM."""

    def __init__(self):
        """Initialize with LLM service."""
        self.llm_service = get_gemini_llm_service()

    @timed_operation(name="llm_query_enhancement", include_args=True, threshold_ms=100)
    async def enhance_query(
        self, original_query: str
    ) -> tuple[list[str], list[str], str]:
        try:
            # Direct LLM call for query enhancement
            prompt = get_enhancement_prompt(original_query)

            # Call LLM service generate_text method
            llm_response = await self.llm_service.generate_text(prompt, temperature=0.3)
            response_text = llm_response.get("text", "")

            # Parse LLM response
            try:
                result = json.loads(response_text)
                concepts = result.get("concepts", [original_query])
                enhanced_queries = result.get("enhanced_queries", [original_query])
                intent_type = result.get("intent_type", "LOOKUP")

                logger.info(
                    f"LLM enhancement successful: {len(enhanced_queries)} queries, intent: {intent_type}"
                )
                return concepts, enhanced_queries, intent_type

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response: {e}")
                # Fallback to original query
                return [original_query], [original_query], "LOOKUP"

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            # Fallback to original query
            return [original_query], [original_query], "LOOKUP"
