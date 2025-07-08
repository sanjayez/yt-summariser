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
