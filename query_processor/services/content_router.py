"""
Routes different content types to appropriate processors
"""

from typing import Dict, Any
from django.core.exceptions import ValidationError

from asgiref.sync import sync_to_async
from ..models import QueryRequest
from ..validators import validate_request_content
from .query_processing import QueryProcessor
from telemetry.logging.logger import get_logger

logger = get_logger(__name__)


class ContentRouterService:
    """
    Service for routing different content types to appropriate processors.
    
    This service acts as the main entry point for all content processing requests,
    handling validation, database operations, and routing to the correct processors.
    """
    
    @staticmethod
    async def route_request(unified_session, request_type: str, content: str) -> Dict[str, Any]:
        """
        Route request to appropriate processing workflow asynchronously.
        
        Args:
            unified_session: UnifiedSession instance
            request_type: Type of request ('video', 'playlist', 'topic')
            content: User content (URL or query)
            
        Returns:
            dict: Processing result with request info
            
        Raises:
            ValidationError: If content validation fails
            Exception: If routing or processing fails
        """
        logger.info(f"Routing {request_type} request for session {unified_session.session_id}")
        
        try:
            # Validate content based on request type
            validation_result = validate_request_content(content, request_type)
            logger.debug(f"Content validation successful for {request_type} request")
            
            # Create QueryRequest record (async-safe)
            query_request = await sync_to_async(QueryRequest.objects.create)(
                unified_session=unified_session,
                request_type=request_type,
                **validation_result,
                status='processing'
            )
            logger.info(f"Created QueryRequest {query_request.search_id} for {request_type}")
            
            # Route to appropriate processor
            if request_type in ['video', 'playlist']:
                return await ContentRouterService._route_video_request(query_request)
            elif request_type == 'topic':
                return await ContentRouterService._route_topic_request(query_request)
            else:
                raise ValueError(f"Unsupported request type: {request_type}")
                
        except ValidationError as e:
            logger.warning(f"Content validation failed for {request_type}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Request routing failed for {request_type}: {str(e)}")
            raise
    
    @staticmethod
    async def _route_video_request(query_request: QueryRequest) -> Dict[str, Any]:
        try:
            logger.info(f"Routing {query_request.request_type} request {query_request.search_id}")
            
            # For video/playlist: Just update QueryRequest status to success
            # No LLM enhancement or search needed - it's a direct URL
            await sync_to_async(lambda: setattr(query_request, 'status', 'success'))()
            await sync_to_async(query_request.save)()
            
            logger.info(f"Successfully processed {query_request.request_type} request {query_request.search_id}")
            
            return {
                'search_id': str(query_request.search_id),
                'type': query_request.request_type,
                'status': 'success',
                'message': f'{query_request.request_type.title()} request processed successfully',
                'url': query_request.video_urls[0] if query_request.video_urls else query_request.original_content,
                'video_urls': query_request.video_urls,
                'total_videos': query_request.total_videos
            }
            
        except Exception as e:
            error_msg = f"Failed to route {query_request.request_type} request: {str(e)}"
            logger.error(error_msg)
            
            # Update QueryRequest with error
            await ContentRouterService._update_query_request_error(query_request, error_msg)
            
            raise Exception(error_msg) from e
    
    @staticmethod
    async def _route_topic_request(query_request: QueryRequest) -> Dict[str, Any]:
        try:
            logger.info(f"Routing topic request {query_request.search_id}")
            
            processor = QueryProcessor()

            result = await processor.process_query_request(query_request)
            
            logger.info(f"Completed topic processing for {query_request.search_id}: {result.get('status')}")
            
            return {
                'search_id': str(query_request.search_id),
                'type': 'topic',
                'status': result.get('status', 'completed'),
                'message': 'Topic query processing completed',
                'query': query_request.original_content,
                'concepts': result.get('concepts', []),
                'enhanced_queries': result.get('enhanced_queries', []),
                'intent_type': result.get('intent_type', ''),
                'video_urls': result.get('video_urls', []),
                'total_videos': result.get('total_videos', 0)
            }
            
        except Exception as e:
            error_msg = f"Failed to route topic request: {str(e)}"
            logger.error(error_msg)
            
            # Update QueryRequest with error
            await ContentRouterService._update_query_request_error(query_request, error_msg)
            
            raise Exception(error_msg) from e
    
    @staticmethod
    async def _update_query_request_error(query_request: QueryRequest, error_message: str):
        try:
            await sync_to_async(lambda: setattr(query_request, 'status', 'failed'))()
            await sync_to_async(lambda: setattr(query_request, 'error_message', error_message))()
            await sync_to_async(query_request.save)()
            logger.debug(f"Updated QueryRequest {query_request.search_id} with error status")
        except Exception as e:
            logger.error(f"Failed to update QueryRequest error status: {str(e)}")
    
