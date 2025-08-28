"""
Query Processing Service
Simple, performant query processing with LLM enhancement and YouTube search
"""

import asyncio
from typing import Dict, Any, List
from django.db import transaction

from asgiref.sync import sync_to_async
from ..models import QueryRequest
from .query_enhancer import QueryEnhancementService
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from video_processor.config import BUSINESS_LOGIC_CONFIG
from telemetry.logging.logger import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """Simple query processor with LLM enhancement and YouTube search."""
    
    def __init__(self):
        """Initialize query processor with default services."""
        self.enhancement_service = QueryEnhancementService()
        self.search_provider = ScrapeTubeProvider(
            max_results=5,
            timeout=30,
            filter_shorts=True,
            english_only=True,
            min_duration_seconds=BUSINESS_LOGIC_CONFIG['DURATION_LIMITS']['minimum_seconds'],
            max_duration_seconds=BUSINESS_LOGIC_CONFIG['DURATION_LIMITS']['maximum_seconds']
        )
    
    async def process_query_request(self, query_request: QueryRequest, max_videos: int = 5) -> Dict[str, Any]:
        """Process query request with LLM enhancement and YouTube search."""
        try:
            # Enhance query with LLM (using dedicated service)
            logger.info(f"🔍 Processing original query: '{query_request.original_content}'")
            concepts, enhanced_queries, intent_type = await self.enhancement_service.enhance_query(
                query_request.original_content
            )
            
            # Search YouTube for videos
            video_urls = await self._search_videos(enhanced_queries, max_videos)
            logger.info(f"🎥 Found {len(video_urls)} videos: {video_urls[:3]}{'...' if len(video_urls) > 3 else ''}")
            
            # Update database (async-safe)
            await sync_to_async(self._update_query_request)(
                query_request, concepts, enhanced_queries, intent_type, video_urls
            )
            
            # For this PR scope: No URLRequestTable entries needed
            url_request_ids = []
            
            return {
                'status': 'success',
                'search_id': str(query_request.search_id),
                'concepts': concepts,
                'enhanced_queries': enhanced_queries,
                'intent_type': intent_type,
                'video_urls': video_urls,
                'total_videos': len(video_urls),
                'url_request_ids': url_request_ids
            }
            
        except Exception as e:
            logger.error(f"Query processing failed for {query_request.search_id}: {e}")
            
            # Update query request with error (async-safe)
            await sync_to_async(self._update_query_request_error)(query_request, str(e))
            
            return {
                'status': 'failed',
                'search_id': str(query_request.search_id),
                'error': str(e)
            }
    

    async def _search_videos(self, queries: List[str], max_videos: int) -> List[str]:
        """Search YouTube for videos using enhanced queries."""
        # TODO: Replace with Node.js sidecar HTTP call for better performance
        # async with aiohttp.ClientSession() as session:
        #     tasks = [
        #         session.post('/api/search', json={'query': query, 'max_results': max_videos})
        #         for query in queries[:3]
        #     ]
        #     results = await asyncio.gather(*tasks)
        
        async def search_query(query: str) -> List[str]:
            """Search for a single query using ScrapeTube (current implementation)."""
            try:
                # Use asyncio.to_thread for cleaner async execution (Python 3.9+)
                return await asyncio.to_thread(self.search_provider.search, query)
            except Exception as e:
                logger.error(f"Search failed for '{query}': {e}")
                return []
        
        # Run searches concurrently (limit to 3 for rate limiting)
        tasks = [search_query(query) for query in queries[:3]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect unique URLs
        seen_urls = set()
        video_urls = []
        
        for result in results:
            if isinstance(result, list):
                for url in result:
                    if url not in seen_urls and len(video_urls) < max_videos:
                        seen_urls.add(url)
                        video_urls.append(url)
        
        return video_urls
    
    def _update_query_request(self, query_request: QueryRequest, concepts: List[str], 
                             enhanced_queries: List[str], intent_type: str, video_urls: List[str]):
        """Update QueryRequest with processing results."""
        with transaction.atomic():
            query_request.concepts = concepts
            query_request.enhanced_queries = enhanced_queries
            query_request.intent_type = intent_type
            query_request.video_urls = video_urls
            query_request.total_videos = len(video_urls)
            query_request.status = 'success'
            query_request.save()
    
    def _update_query_request_error(self, query_request: QueryRequest, error_message: str):
        """Update QueryRequest with error status."""
        with transaction.atomic():
            query_request.status = 'failed'
            query_request.error_message = error_message
            query_request.save()
