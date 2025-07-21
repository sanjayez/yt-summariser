"""
Celery Tasks for Topic Search Processing
Handles asynchronous YouTube search and AI processing operations
"""

import logging
import asyncio
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction

from ai_utils.config import get_config
from ai_utils.providers.openai_llm import OpenAILLMProvider
from ai_utils.services.llm_service import LLMService
from topic.services.query_processor import QueryProcessor
from topic.services.search_service import YouTubeSearchService, SearchRequest
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from topic.utils.session_utils import update_session_status
from topic.models import SearchRequest as SearchRequestModel, SearchSession
from video_processor.config import YOUTUBE_CONFIG, BUSINESS_LOGIC_CONFIG
from video_processor.utils import handle_dead_letter_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, 
             name='topic.process_search_query',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['search_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['search_hard_limit'],
             max_retries=YOUTUBE_CONFIG['RETRY_CONFIG']['search']['max_retries'],
             default_retry_delay=YOUTUBE_CONFIG['RETRY_CONFIG']['search']['countdown'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['search']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['search']['jitter'])
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
    
    search_request = None
    session = None
    
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
                min_duration_seconds=BUSINESS_LOGIC_CONFIG['DURATION_LIMITS']['minimum_seconds'],
                max_duration_seconds=BUSINESS_LOGIC_CONFIG['DURATION_LIMITS']['maximum_seconds']
            )
            youtube_search_service = YouTubeSearchService(youtube_search_provider)
            
            logger.debug("AI services initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize AI services: {str(e)}"
            logger.error(error_msg)
            
            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)
            
            return {
                'status': 'failed',
                'error': 'AI services initialization failed',
                'details': str(e),
                'search_id': search_id
            }
        
        # Process query with LLM enhancement (async operation)
        try:
            logger.debug("Starting LLM query processing")
            
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
                loop = asyncio.get_event_loop()
            
            # Process query asynchronously - use enhance_query method
            enhancement_result = loop.run_until_complete(
                query_processor.enhance_query(original_query)
            )
            
            # Extract enhanced query from result
            if enhancement_result.get("status") == "completed":
                processed_query = enhancement_result.get("enhanced_query", original_query)
                logger.info(f"Enhanced query: '{original_query}' → '{processed_query}'")
            else:
                processed_query = original_query
                logger.warning(f"Query enhancement failed: {enhancement_result.get('error', 'Unknown error')}, using original query")
            
        except Exception as e:
            error_msg = f"Query enhancement failed: {str(e)}"
            logger.error(error_msg)
            
            # Use original query if enhancement fails
            processed_query = original_query
            logger.info("Using original query due to enhancement failure")
        
        # Perform YouTube search
        try:
            logger.debug(f"Starting YouTube search for: '{processed_query}'")
            
            search_request_obj = SearchRequest(
                query=processed_query,
                max_results=max_videos
            )
            
            search_response = youtube_search_service.search(search_request_obj)
            video_urls = search_response.results
            
            logger.info(f"Found {len(video_urls)} videos for query: '{processed_query}'")
            
            if not video_urls:
                logger.warning(f"No videos found for query: '{processed_query}'")
                
        except Exception as e:
            error_msg = f"YouTube search failed: {str(e)}"
            logger.error(error_msg)
            
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
        
    except SoftTimeLimitExceeded:
        # Search processing is approaching timeout
        logger.warning(f"Search processing soft timeout reached for search {search_id}")
        
        try:
            # Mark search as failed due to timeout
            if search_request:
                _update_search_request_error(search_request, "Search processing timed out - query may be too complex")
            if session:
                _update_session_error(session)
            logger.error(f"Marked search processing as failed due to timeout: {search_id}")
            
        except Exception as cleanup_error:
            logger.error(f"Failed to update search status during timeout cleanup: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Search processing timeout for search {search_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error in search processing: {e}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('process_search_query', self.request.id, [search_id], {}, e)
        
        try:
            if search_request:
                _update_search_request_error(search_request, f"Unexpected error: {str(e)}")
            if session:
                _update_session_error(session)
        except:
            pass
            
        return {
            'status': 'failed',
            'error': 'Unexpected processing error',
            'details': str(e),
            'search_id': search_id
        }


@shared_task(bind=True, 
             name='topic.process_search_with_videos',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['parallel_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['parallel_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['parallel']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['parallel']['jitter'])
def process_search_with_videos(self, search_id: str, max_videos: int = 5, start_video_processing: bool = False):
    """
    Combined search and video processing workflow.
    
    Args:
        search_id: UUID of the SearchRequest to process
        max_videos: Maximum number of videos to find and process
        start_video_processing: Whether to immediately start video processing
        
    Returns:
        dict: Combined search and processing result
    """
    logger.info(f"Starting combined search and video processing for {search_id}")
    
    search_request = None
    
    try:
        # First, run the search
        search_result = process_search_query(search_id, max_videos)
        
        if search_result['status'] != 'success':
            return search_result
        
        # If video processing is requested, start it
        if start_video_processing and search_result.get('video_urls'):
            logger.info(f"Starting video processing for {len(search_result['video_urls'])} videos")
            
            try:
                # Import here to avoid circular imports
                from topic.parallel_tasks import process_search_results
                
                # Start video processing
                processing_task = process_search_results.delay(search_id)
                
                search_result['video_processing_task_id'] = processing_task.id
                search_result['video_processing_started'] = True
                
            except Exception as e:
                logger.error(f"Failed to start video processing: {e}")
                search_result['video_processing_error'] = str(e)
                search_result['video_processing_started'] = False
        
        return search_result
        
    except SoftTimeLimitExceeded:
        # Combined processing is approaching timeout
        logger.warning(f"Combined search and video processing soft timeout reached for search {search_id}")
        
        try:
            # Mark search as failed due to timeout
            if search_request:
                _update_search_request_error(search_request, "Combined processing timed out")
            logger.error(f"Marked combined processing as failed due to timeout: {search_id}")
            
        except Exception as cleanup_error:
            logger.error(f"Failed to update status during combined processing timeout cleanup: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Combined search and video processing timeout for search {search_id}")
        
    except Exception as e:
        logger.error(f"Error in combined search and video processing: {e}")
        return {
            'status': 'failed',
            'error': 'Combined processing failed',
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
        logger.error(f"Failed to update session with error: {e}") 