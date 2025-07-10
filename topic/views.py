from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError, APIException
from django.db import transaction
import logging

from topic.utils.session_utils import get_or_create_session
from topic.models import SearchRequest as SearchRequestModel
from topic.serializers import TopicSearchRequestSerializer, TopicSearchResponseSerializer
from topic.tasks import process_search_query
from api.models import URLRequestTable
from api.serializers import URLRequestTableSerializer
from api.utils.get_client_ip import get_client_ip
from video_processor.validators import validate_youtube_url
from video_processor.processors.workflow import process_youtube_video

logger = logging.getLogger(__name__)


class TopicSearchAPIView(APIView):
    """
    Search for YouTube videos based on a user query.
    
    This endpoint processes user queries through:
    1. Session management (get or create session)
    2. Query enhancement using LLM
    3. YouTube search using ScrapeTube
    4. Database persistence of results
    
    Request Body:
        {
            "query": "string (required) - User's search query"
        }
    
    Response:
        {
            "session_id": "uuid-here", 
            "original_query": "user query",
            "status": "processing"
        }
    """
    
    def post(self, request):
        """Handle POST requests for topic search"""
        search_request = None
        
        try:
            # Validate request data using serializer
            request_serializer = TopicSearchRequestSerializer(data=request.data)
            if not request_serializer.is_valid():
                raise ValidationError(request_serializer.errors)
            
            query = request_serializer.validated_data['query']
            logger.info(f"Received topic search request for query: '{query}'")
            
            # Get or create session for this request
            try:
                session = get_or_create_session(request)
                logger.debug(f"Using session {session.session_id} for search request")
            except Exception as e:
                logger.error(f"Session creation failed: {e}")
                raise APIException(f"Session creation failed: {str(e)}")
            
            # Create search request record
            try:
                with transaction.atomic():
                    search_request = SearchRequestModel.objects.create(
                        search_session=session,
                        original_query=query,
                        status='processing'
                    )
                    logger.debug(f"Created search request {search_request.request_id}")
            except Exception as e:
                logger.error(f"Failed to create search request: {e}")
                raise APIException("Failed to create search request")
            
            # Dispatch Celery task for async processing
            try:
                task = process_search_query.delay(str(search_request.request_id))
                logger.info(f"Dispatched async task {task.id} for search request {search_request.request_id}")
            except Exception as e:
                logger.error(f"Failed to dispatch Celery task: {e}")
                # Update search request status
                search_request.status = 'failed'
                search_request.error_message = f"Failed to dispatch processing task: {str(e)}"
                search_request.save()
                raise APIException("Failed to start background processing")
            
            # Prepare immediate response
            response_data = {
                'session_id': str(session.session_id),
                'original_query': query,
                'status': 'processing'
            }
            
            # Validate response data using serializer
            response_serializer = TopicSearchResponseSerializer(data=response_data)
            if response_serializer.is_valid():
                logger.info(f"Topic search initiated for query: '{query}' with task {task.id}")
                return Response(response_serializer.validated_data, status=status.HTTP_202_ACCEPTED)
            else:
                logger.error(f"Response validation failed: {response_serializer.errors}")
                raise APIException("Response validation failed")
                
        except ValidationError:
            # Re-raise validation errors as-is (DRF will handle them)
            raise
            
        except APIException:
            # Re-raise API exceptions as-is (DRF will handle them)
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in search_topic: {e}")
            
            # Update database with error if search_request exists
            if search_request:
                try:
                    search_request.status = 'failed'
                    search_request.error_message = f"Unexpected error: {str(e)}"
                    search_request.save()
                except Exception as db_error:
                    logger.error(f"Failed to update search request with error: {db_error}")
            
            raise APIException(f"Internal server error: {str(e)}")


class SearchAndProcessAPIView(APIView):
    """
    Combined search and video processing endpoint.
    
    This endpoint:
    1. Performs a topic search to find videos
    2. Creates URLRequestTable entries for each video
    3. Links them to the search request
    4. Triggers video processing for each video
    
    Request Body:
        {
            "query": "string (required) - User's search query",
            "max_videos": "integer (optional) - Maximum number of videos to process (default: 3)"
        }
    
    Response:
        {
            "session_id": "uuid",
            "search_request_id": "uuid", 
            "original_query": "string",
            "status": "processing",
            "video_requests": [
                {
                    "request_id": "uuid",
                    "url": "string",
                    "status": "processing"
                }
            ]
        }
    """
    
    def post(self, request):
        """Process search query and start video processing"""
        try:
            # Get and validate input
            query = request.data.get('query', '').strip()
            max_videos = request.data.get('max_videos', 3)
            
            if not query:
                return Response(
                    {'error': 'Query parameter is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate max_videos
            try:
                max_videos = int(max_videos)
                if max_videos < 1 or max_videos > 10:
                    max_videos = 3
            except (ValueError, TypeError):
                max_videos = 3
                
            # Get client IP
            ip_address = get_client_ip(request)
            
            # Start database transaction
            with transaction.atomic():
                # Create search session and request
                session = get_or_create_session(request)
                
                search_request = SearchRequestModel.objects.create(
                    search_session=session,
                    original_query=query,
                    status='processing'
                )
                
                logger.info(f"Created search request {search_request.request_id} for query: '{query}'")
                
                # Trigger search processing
                search_task = process_search_query.delay(str(search_request.request_id))
                logger.info(f"Dispatched search task {search_task.id}")
                
                # Wait briefly for search to complete (or timeout)
                # This is a simplified approach - in production you might want to poll
                import time
                time.sleep(2)
                
                # Refresh search request to get results
                search_request.refresh_from_db()
                
                video_requests = []
                if search_request.video_urls:
                    # Process up to max_videos from the search results
                    video_urls = search_request.video_urls[:max_videos]
                    
                    for url in video_urls:
                        try:
                            # Validate YouTube URL
                            validate_youtube_url(url)
                            
                            # Create URLRequestTable entry linked to search request
                            url_request_data = {
                                'url': url,
                                'ip_address': ip_address,
                                'status': 'processing'
                            }
                            
                            url_serializer = URLRequestTableSerializer(data=url_request_data)
                            if url_serializer.is_valid():
                                url_request = url_serializer.save()
                                
                                # Link to search request
                                url_request.search_request = search_request
                                url_request.save()
                                
                                # Trigger video processing
                                process_youtube_video.delay(url_request.id)
                                
                                video_requests.append({
                                    'request_id': str(url_request.request_id),
                                    'url': url,
                                    'status': 'processing'
                                })
                                
                                logger.info(f"Created video request {url_request.request_id} for URL: {url}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to process URL {url}: {e}")
                            continue
                
                # Update search request with video count
                search_request.total_videos = len(video_requests)
                search_request.save()
                
                # Prepare response
                response_data = {
                    'session_id': str(session.session_id),
                    'search_request_id': str(search_request.request_id),
                    'original_query': query,
                    'status': 'processing',
                    'video_requests': video_requests,
                    'total_videos': len(video_requests)
                }
                
                return Response(response_data, status=status.HTTP_202_ACCEPTED)
                
        except Exception as e:
            logger.error(f"Error in SearchAndProcessAPIView: {e}")
            return Response(
                {'error': f'Internal server error: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
