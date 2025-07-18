"""
Celery Tasks for Topic Search Processing
Handles asynchronous YouTube search and AI processing operations
"""

import logging
import asyncio
from celery import shared_task
from django.db import transaction

from ai_utils.config import get_config
from ai_utils.providers.openai_llm import OpenAILLMProvider
from ai_utils.services.llm_service import LLMService
from topic.services.query_processor import QueryProcessor
from topic.services.search_service import YouTubeSearchService, SearchRequest
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from topic.utils.session_utils import update_session_status
from topic.models import SearchRequest as SearchRequestModel, SearchSession

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='topic.process_search_query')
def process_search_query(self, search_id: str, max_videos: int = 5):
    """
    Process a search query asynchronously
    
    This task handles:
    1. AI services initialization
    2. LLM query enhancement  
    3. YouTube search
    4. Database updates with results
    
    Args:
        search_id: UUID of the SearchRequest to process
        
    Returns:
        dict: Processing result with status and details
    """
    logger.info(f"Starting async processing for search request: {search_id}")
    
    try:
        # Get search request from database
        try:
            search_request = SearchRequestModel.objects.select_related('search_session').get(
                search_id=search_id
            )
            session = search_request.search_session
            original_query = search_request.original_query
            
            logger.debug(f"Processing search request: {search_id} for query: '{original_query}'")
            
        except SearchRequestModel.DoesNotExist:
            error_msg = f"Search request {search_id} not found"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'Search request not found',
                'search_id': search_id
            }
        
        # Initialize AI services
        try:
            config = get_config()
            config.validate()
            
            llm_provider = OpenAILLMProvider(config=config)
            llm_service = LLMService(provider=llm_provider)

            # get clients for LLM and YouTube search
            query_processor = QueryProcessor(llm_service=llm_service)
            
            # Configure YouTube search service with English-only and shorts filtering
            youtube_search_provider = ScrapeTubeProvider(
                max_results=max_videos,
                timeout=30,
                filter_shorts=True,      # Filter out YouTube shorts
                english_only=True,       # Only English videos
                min_duration_seconds=60, # Videos must be at least 60 seconds
                max_duration_seconds=900 # Cap videos at 15 minutes for MVP testing
            )
            youtube_search_service = YouTubeSearchService(youtube_search_provider)
            
            logger.debug("AI services initialized successfully")
            
        except Exception as e:
            error_msg = f"AI services initialization failed: {str(e)}"
            logger.error(error_msg)
            
            # Update database with error
            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)
            
            return {
                'status': 'failed',
                'error': 'AI services initialization failed',
                'details': str(e),
                'search_id': search_id
            }
        
        # Process query using LLM
        processed_query = original_query  # Fallback to original query
        try:
            logger.debug(f"Enhancing query with LLM: '{original_query}'")
            
            # Run async query enhancement (Celery-safe approach)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                enhancement_result = loop.run_until_complete(
                    query_processor.enhance_query(original_query, job_id=f"search_{search_id}")
                )
            finally:
                loop.close()
            
            if enhancement_result['status'] == 'completed':
                processed_query = enhancement_result['enhanced_query']
                logger.info(f"Query enhanced: '{original_query}' -> '{processed_query}'")
            else:
                logger.warning(f"Query enhancement failed: {enhancement_result.get('error', 'Unknown error')}")
                # Continue with original query as fallback
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Continue with original query as fallback
            processed_query = original_query
        
        # Search YouTube videos
        video_urls = []
        try:
            logger.debug(f"Searching YouTube for: '{processed_query}'")
            
            search_request_obj = SearchRequest(
                query=processed_query,
                max_results=max_videos,
                include_metadata=False
            )
            
            search_response = youtube_search_service.search(search_request_obj)
            video_urls = search_response.results
            
            logger.info(f"Found {len(video_urls)} videos for query '{processed_query}'")
            
        except Exception as e:
            error_msg = f"YouTube search failed: {str(e)}"
            logger.error(error_msg)
            
            # Update database with error
            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)
            
            return {
                'status': 'failed',
                'error': 'YouTube search failed',
                'details': str(e),
                'search_id': search_id
            }
        
        # Update database with results
        try:
            with transaction.atomic():
                search_request.processed_query = processed_query
                search_request.video_urls = video_urls
                search_request.total_videos = len(video_urls)
                search_request.status = 'success'
                search_request.save()
                
                # Update session status to success
                update_session_status(session, 'success')
                
                logger.info(f"Search request {search_id} completed successfully with {len(video_urls)} videos")
                
        except Exception as e:
            error_msg = f"Failed to update search results: {str(e)}"
            logger.error(error_msg)
            
            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)
            
            return {
                'status': 'failed',
                'error': 'Database update failed',
                'details': str(e),
                'search_id': search_id
            }
        
        # Return success result
        return {
            'status': 'success',
            'search_id': search_id,
            'session_id': str(session.session_id),
            'original_query': original_query,
            'processed_query': processed_query,
            'video_urls': video_urls,
            'total_videos': len(video_urls)
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in process_search_query: {e}")
        
        # Try to update database with error if we have the search_request
        if 'search_request' in locals():
            try:
                _update_search_request_error(search_request, f"Unexpected error: {str(e)}")
                _update_session_error(search_request.search_session)
            except Exception as db_error:
                logger.error(f"Failed to update search request with error: {db_error}")
        
        return {
            'status': 'failed',
            'error': 'Unexpected error',
            'details': str(e),
            'search_id': search_id
        }


def _update_search_request_error(search_request: SearchRequestModel, error_message: str):
    """Helper function to update search request with error"""
    try:
        search_request.status = 'failed'
        search_request.error_message = error_message
        search_request.save()
    except Exception as e:
        logger.error(f"Failed to update search request with error: {e}")


def _update_session_error(session: SearchSession):
    """Helper function to update session with error status"""
    try:
        update_session_status(session, 'failed')
    except Exception as e:
        logger.error(f"Failed to update session status: {e}")


@shared_task(bind=True, name='topic.process_search_with_videos')
def process_search_with_videos(self, search_id: str, max_videos: int = 5, start_video_processing: bool = False):
    """
    Integrated search and video processing task.
    
    This task performs the search and optionally starts video processing
    for the found videos.
    
    Args:
        search_id: UUID of the SearchRequest to process
        start_video_processing: Whether to start video processing after search
        
    Returns:
        dict: Processing result with status and details
    """
    logger.info(f"Starting integrated search and video processing for request: {search_id}")
    
    try:
        # First, perform the search - call directly, not async
        search_result = process_search_query(search_id, max_videos)
        
        if search_result['status'] != 'success':
            logger.error(f"Search failed for request {search_id}: {search_result}")
            return search_result
        
        # If video processing is requested and we have videos
        if start_video_processing and search_result.get('video_urls'):
            logger.info(f"Starting video processing for {len(search_result['video_urls'])} videos")
            
            # Import here to avoid circular imports
            from topic.parallel_tasks import process_search_results
            
            # Start parallel video processing
            video_processing_result = process_search_results.apply_async(args=[search_id])
            
            # Return combined result
            return {
                'status': 'processing_videos',
                'search_id': search_id,
                'search_result': search_result,
                'video_processing_task_id': video_processing_result.id,
                'message': f"Search completed, processing {len(search_result['video_urls'])} videos"
            }
        else:
            # Return search result only
            return search_result
            
    except Exception as e:
        logger.error(f"Unexpected error in process_search_with_videos: {e}")
        return {
            'status': 'failed',
            'error': 'Unexpected error',
            'details': str(e),
            'search_id': search_id
        } 