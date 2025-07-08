from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError, APIException
from django.db import transaction
import logging
import asyncio
import time

from ai_utils.config import get_config
from ai_utils.providers.openai_llm import OpenAILLMProvider
from ai_utils.services.llm_service import LLMService
from topic.services.query_processor import QueryProcessor
from topic.services.search_service import YouTubeSearchService, SearchRequest
from topic.utils.session_utils import get_or_create_session, update_session_status
from topic.models import SearchRequest as SearchRequestModel
from topic.serializers import TopicSearchRequestSerializer, TopicSearchResponseSerializer

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
            "search_request_id": "uuid-here",
            "session_id": "uuid-here", 
            "original_query": "user query",
            "processed_query": "enhanced query",
            "video_urls": ["https://youtube.com/watch?v=...", ...],
            "total_videos": 5,
            "processing_time_ms": 1234.56,
            "status": "success"
        }
    """
    
    def post(self, request):
        """Handle POST requests for topic search"""
        start_time = time.time()
        search_request = None
        
        try:
            # Validate request data using serializer
            request_serializer = TopicSearchRequestSerializer(data=request.data)
            if not request_serializer.is_valid():
                raise ValidationError(request_serializer.errors)
            
            query = request_serializer.validated_data['query']
            logger.info(f"Processing topic search request for query: '{query}'")
            
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
            
            # Initialize AI services
            try:
                config = get_config()
                config.validate()
                
                llm_provider = OpenAILLMProvider(config=config)
                llm_service = LLMService(provider=llm_provider)
                query_processor = QueryProcessor(llm_service=llm_service)
                youtube_search_service = YouTubeSearchService()
                
                logger.debug("AI services initialized successfully")
            except Exception as e:
                logger.error(f"AI services initialization failed: {e}")
                
                # Update database with error
                if search_request:
                    search_request.status = 'failed'
                    search_request.error_message = f"AI services initialization failed: {str(e)}"
                    search_request.save()
                
                raise APIException('Failed to initialize AI services for query processing')
            
            # Process query using LLM
            processed_query = query  # Fallback to original query
            try:
                logger.debug(f"Enhancing query with LLM: '{query}'")
                
                # Run async query enhancement
                enhancement_result = asyncio.run(
                    query_processor.enhance_query(query, job_id=f"search_{search_request.request_id}")
                )
                
                if enhancement_result['status'] == 'completed':
                    processed_query = enhancement_result['enhanced_query']
                    logger.info(f"Query enhanced: '{query}' -> '{processed_query}'")
                else:
                    logger.warning(f"Query enhancement failed: {enhancement_result.get('error', 'Unknown error')}")
                    # Continue with original query as fallback
                    
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
                # Continue with original query as fallback
                processed_query = query
            
            # Search YouTube videos
            video_urls = []
            try:
                logger.debug(f"Searching YouTube for: '{processed_query}'")
                
                search_request_obj = SearchRequest(
                    query=processed_query,
                    max_results=5,
                    include_metadata=False
                )
                
                search_response = youtube_search_service.search(search_request_obj)
                video_urls = search_response.results
                
                logger.info(f"Found {len(video_urls)} videos for query '{processed_query}'")
                
            except Exception as e:
                logger.error(f"YouTube search failed: {e}")
                
                # Update database with error
                search_request.status = 'failed'
                search_request.error_message = f"YouTube search failed: {str(e)}"
                search_request.save()
                
                raise APIException(f"YouTube search failed: {str(e)}")
            
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
                    
                    logger.info(f"Search request {search_request.request_id} completed successfully")
                    
            except Exception as e:
                logger.error(f"Failed to update search request: {e}")
                raise APIException("Failed to save search results")
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Prepare response data
            response_data = {
                'search_request_id': str(search_request.request_id),
                'session_id': str(session.session_id),
                'original_query': query,
                'processed_query': processed_query,
                'video_urls': video_urls,
                'total_videos': len(video_urls),
                'processing_time_ms': round(processing_time_ms, 2),
                'status': 'success'
            }
            
            # Validate response data using serializer
            response_serializer = TopicSearchResponseSerializer(data=response_data)
            if response_serializer.is_valid():
                logger.info(f"Topic search completed in {processing_time_ms:.2f}ms for query: '{query}'")
                return Response(response_serializer.validated_data, status=status.HTTP_200_OK)
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
