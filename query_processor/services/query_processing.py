"""
Query Processing Service
Simple, performant query processing with LLM enhancement and YouTube search
"""

import asyncio
from typing import Any

from asgiref.sync import sync_to_async
from django.db import transaction

from telemetry.logging.logger import get_logger
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from video_processor.config import BUSINESS_LOGIC_CONFIG

from ..models import QueryRequest
from .query_enhancer import QueryEnhancementService

logger = get_logger(__name__)


class QueryProcessor:
    """Simple query processor with LLM enhancement and YouTube search."""

    def __init__(
        self,
        enhancement_service: "QueryEnhancementService" = None,
        search_provider: "ScrapeTubeProvider" = None,
    ):
        """Initialize query processor with default services."""
        self.enhancement_service = enhancement_service or QueryEnhancementService()

        # Safe config access with sensible fallbacks
        limits = BUSINESS_LOGIC_CONFIG.get("DURATION_LIMITS", {}) or {}  # type: ignore[attr-defined]
        min_seconds = limits.get("minimum_seconds", 60)  # type: ignore[attr-defined]
        max_seconds = limits.get("maximum_seconds", None)  # type: ignore[attr-defined]

        self.search_provider = search_provider or ScrapeTubeProvider(
            max_results=5,
            timeout=30,
            filter_shorts=True,
            english_only=True,
            min_duration_seconds=min_seconds,
            max_duration_seconds=max_seconds,
        )

    async def process_query_request(
        self, query_request: QueryRequest, max_videos: int = 5
    ) -> dict[str, Any]:
        """Process query request with LLM enhancement and YouTube search."""
        try:
            # Validate and clamp max_videos to prevent resource exhaustion
            max_videos = max(1, min(int(max_videos or 5), 20))

            (
                concepts,
                enhanced_queries,
                intent_type,
            ) = await self.enhancement_service.enhance_query(
                query_request.original_content, model="gemini-2.5-flash-lite"
            )

            # Search YouTube for videos
            video_urls = await self._search_videos(enhanced_queries, max_videos)
            logger.info(
                f"🎥 Found {len(video_urls)} videos: {video_urls[:3]}{'...' if len(video_urls) > 3 else ''}"
            )

            # Update database (async-safe)
            await sync_to_async(self._update_query_request)(
                query_request, concepts, enhanced_queries, intent_type, video_urls
            )

            # For this PR scope: No URLRequestTable entries needed
            url_request_ids: list[str] = []

            return {
                "status": "success",
                "search_id": str(query_request.search_id),
                "concepts": concepts,
                "enhanced_queries": enhanced_queries,
                "intent_type": intent_type,
                "video_urls": video_urls,
                "total_videos": len(video_urls),
                "url_request_ids": url_request_ids,
            }

        except Exception as e:
            # Truncate error message to prevent DB bloat
            err_msg = str(e)[:1000]
            # Log full traceback for debugging - critical for production triage
            logger.exception(
                "Query processing failed for %s: %s", query_request.search_id, err_msg
            )

            # Update query request with error (async-safe)
            await sync_to_async(self._update_query_request_error)(
                query_request, err_msg
            )

            return {
                "status": "failed",
                "search_id": str(query_request.search_id),
                "error": err_msg,
            }

    async def _search_videos(self, queries: list[str], max_videos: int) -> list[str]:
        """Search YouTube for videos using enhanced queries."""
        # TODO: Replace with Node.js sidecar HTTP call for better performance
        # async with aiohttp.ClientSession() as session:
        #     tasks = [
        #         session.post('/api/search', json={'query': query, 'max_results': max_videos})
        #         for query in queries[:3]
        #     ]
        #     results = await asyncio.gather(*tasks)

        # Normalize and deduplicate queries while preserving order
        seen_queries = set()
        unique_queries: list[str] = []
        for q in queries or []:
            qn = (q or "").strip()
            if qn and qn not in seen_queries:
                seen_queries.add(qn)
                unique_queries.append(qn)

        if not unique_queries:
            return []

        # Limit concurrent searches for rate limiting
        limited_queries = unique_queries[:3]

        async def search_query(query: str) -> list[str]:
            """Search for a single query using ScrapeTube (current implementation)."""
            try:
                # Calculate per-call limit and add timeout protection
                per_call_max = max(1, min(max_videos, 10))
                return await asyncio.wait_for(
                    asyncio.to_thread(self.search_provider.search, query, per_call_max),
                    timeout=8.0,
                )
            except Exception as e:
                logger.error("Search failed for '%s': %s", query, e, exc_info=True)
                return []

        # Run searches concurrently with exception handling
        tasks = [search_query(query) for query in limited_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and deduplicate URLs from all successful searches
        all_urls = []
        for result in results:
            if isinstance(result, list):  # Successful search result
                all_urls.extend(result)

        # Deduplicate while preserving order and respecting max_videos limit
        seen_urls = set()
        video_urls: list[str] = []
        for url in all_urls:
            if url not in seen_urls and len(video_urls) < max_videos:
                seen_urls.add(url)
                video_urls.append(url)

        return video_urls

    def _update_query_request(
        self,
        query_request: QueryRequest,
        concepts: list[str],
        enhanced_queries: list[str],
        intent_type: str,
        video_urls: list[str],
    ):
        """Update QueryRequest with processing results."""
        with transaction.atomic():
            # Use select_for_update to ensure row lock and prevent race conditions
            obj = QueryRequest.objects.select_for_update().get(pk=query_request.pk)
            obj.concepts = concepts
            obj.enhanced_queries = enhanced_queries
            obj.intent_type = intent_type
            obj.video_urls = video_urls
            obj.total_videos = len(video_urls)
            obj.status = "success"
            obj.error_message = (
                ""  # Clear any previous error message for clean success state
            )
            # Use update_fields for performance - only update changed fields
            obj.save(
                update_fields=[
                    "concepts",
                    "enhanced_queries",
                    "intent_type",
                    "video_urls",
                    "total_videos",
                    "status",
                    "error_message",
                ]
            )

    def _update_query_request_error(
        self, query_request: QueryRequest, error_message: str
    ):
        """Update QueryRequest with error status."""
        with transaction.atomic():
            # Use select_for_update to ensure row lock and prevent race conditions
            obj = QueryRequest.objects.select_for_update().get(pk=query_request.pk)
            obj.status = "failed"  # Using string literal as model doesn't expose STATUS constants
            obj.error_message = (error_message or "")[
                :1000
            ]  # Truncate to prevent field overflow
            # Reset video fields to ensure clean error state
            obj.video_urls = []
            obj.total_videos = 0
            # Use update_fields for performance - only update changed fields
            obj.save(
                update_fields=["status", "error_message", "video_urls", "total_videos"]
            )
