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
    
    @staticmethod
    async def route_request(unified_session, request_type: str, content: str) -> Dict[str, Any]:
        logger.info(f"🔀 Routing {request_type} request for session {unified_session.session_id}")
        
        try:
            # Validate content based on request type
            validation_result = validate_request_content(content, request_type)
            
            # Create QueryRequest record (async-safe)
            query_request = await sync_to_async(QueryRequest.objects.create)(
                unified_session=unified_session,
                request_type=request_type,
                **validation_result,
                status='processing'
            )
            
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
            # For video/playlist: Just update QueryRequest status to success
            # No LLM enhancement or search needed - it's a direct URL
            await sync_to_async(lambda: setattr(query_request, 'status', 'success'))()
            await sync_to_async(query_request.save)()
            
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
            
            # Update QueryRequest with error
            await ContentRouterService._update_query_request_error(query_request, error_msg)
            
            raise Exception(error_msg) from e
    
    @staticmethod
    async def _route_topic_request(query_request: QueryRequest) -> Dict[str, Any]:
        try:
            
            processor = QueryProcessor()

            result = await processor.process_query_request(query_request)
            
            return {
                'search_id': str(query_request.search_id),
                'type': 'topic',
                'status': result.get('status', 'success'),
                'message': (
                    'Topic query processing completed'
                    if result.get('status') == 'success'
                    else f"Topic query processing failed: {result.get('error', '')[:200]}"
                ),
                'query': query_request.original_content,
                'concepts': result.get('concepts', []),
                'enhanced_queries': result.get('enhanced_queries', []),
                'intent_type': result.get('intent_type', ''),
                'video_urls': result.get('video_urls', []),
                'total_videos': result.get('total_videos', 0),
                'error': result.get('error')
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
            err = (error_message or '')[:1000]
            query_request.status = 'failed'
            query_request.error_message = err
            query_request.video_urls = []
            query_request.total_videos = 0
            await sync_to_async(query_request.save)(
                update_fields=['status', 'error_message', 'video_urls', 'total_videos']
            )
            logger.debug("Updated QueryRequest %s with error status", query_request.search_id)
        except Exception:
            logger.exception(
                "Failed to update QueryRequest error status for %s",
                query_request.search_id
            )
    
