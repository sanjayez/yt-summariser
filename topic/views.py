from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.exceptions import ValidationError, APIException
from django.db import transaction
from django.db.models import Q, Count, Case, When
from django.http import StreamingHttpResponse
import logging
import json
import time
from celery_progress.backend import Progress
from celery.result import AsyncResult
import asyncio

from topic.utils.session_utils import get_or_create_session
from topic.models import SearchRequest as SearchRequestModel
from topic.serializers import TopicSearchRequestSerializer, TopicSearchResponseSerializer
from topic.tasks import process_search_query, process_search_with_videos
from topic.parallel_tasks import process_search_results, get_search_processing_status
from api.models import URLRequestTable
from api.serializers import URLRequestTableSerializer
from api.utils.get_client_ip import get_client_ip
from video_processor.validators import validate_youtube_url
from video_processor.processors.workflow import process_youtube_video
from ai_utils.search_process_serializers import (
    SearchToProcessRequestSerializer,
    SearchToProcessResponseSerializer,
    StatusCheckRequestSerializer,
    StatusCheckResponseSerializer
)

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
                    logger.debug(f"Created search request {search_request.search_id}")
            except Exception as e:
                logger.error(f"Failed to create search request: {e}")
                raise APIException("Failed to create search request")
            
            # Dispatch Celery task for async processing
            try:
                task = process_search_query.delay(str(search_request.search_id))
                logger.info(f"Dispatched async task {task.id} for search request {search_request.search_id}")
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
            "search_id": "uuid", 
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
                
                logger.info(f"Created search request {search_request.search_id} for query: '{query}'")
                
                # Trigger search processing
                search_task = process_search_query.delay(str(search_request.search_id))
                logger.info(f"Dispatched search task {search_task.id}")
                
                # Refresh search request to get initial results
                # Note: Search processing happens asynchronously, so initial video_urls may be empty
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
                                process_youtube_video.delay(str(url_request.request_id))
                                
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
                    'search_id': str(search_request.search_id),
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


class ParallelSearchProcessAPIView(APIView):
    """
    Enhanced parallel search and video processing endpoint.
    
    This endpoint provides true parallel processing of multiple videos
    using Celery groups for maximum efficiency.
    """
    
    def post(self, request):
        """Process search query and start parallel video processing"""
        try:
            # Validate request using Pydantic serializer
            serializer = SearchToProcessRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            validated_data = serializer.validated_data
            query = validated_data['query']
            max_videos = validated_data.get('max_videos', 5)
            process_videos = validated_data.get('process_videos', True)
            
            logger.info(f"Starting parallel search and process for query: '{query}'")
            
            # Create session  
            session = get_or_create_session(request)
            
            # Create search request
            with transaction.atomic():
                search_request = SearchRequestModel.objects.create(
                    search_session=session,
                    original_query=query,
                    status='processing'
                )
                
                logger.info(f"Created search request {search_request.search_id}")
                
                # Start processing based on mode
                if process_videos:
                    # Use integrated workflow
                    task = process_search_with_videos.delay(
                        str(search_request.search_id),
                        max_videos
                    )
                    logger.info(f"Started integrated search and video processing task {task.id}")
                else:
                    # Search only
                    task = process_search_query.delay(str(search_request.search_id))
                    logger.info(f"Started search-only task {task.id}")
                
                # Prepare response
                response_data = {
                    'search_id': str(search_request.search_id),
                    'session_id': str(session.session_id),
                    'query': query,
                    'max_videos': max_videos,
                    'process_videos': process_videos,
                    'status': 'processing',
                    'task_id': task.id
                }
                
                # Serialize response
                response_serializer = SearchToProcessResponseSerializer(data=response_data)
                if response_serializer.is_valid():
                    return Response(response_serializer.validated_data, status=status.HTTP_202_ACCEPTED)
                else:
                    return Response(response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Error in ParallelSearchProcessAPIView: {e}")
            return Response(
                {'error': f'Internal server error: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


async def simple_sse_test(request):
    """
    Async SSE test endpoint for proper ASGI streaming - uses async generators
    """
    async def event_stream():
        import json
        
        logger.info("Starting async simple SSE test stream")
        
        # Send initial connection message with padding to prevent buffering
        yield f"data: {json.dumps({'type': 'connected', 'message': 'TESTING-123-UNIQUE-ASYNC-SSE-MESSAGE', 'timestamp': time.time()})}\n\n"
        yield f": padding to prevent buffering {' ' * 1024}\n\n"  # Force flush with padding
        
        # Send 10 messages with 2-second delays using async sleep
        for i in range(10):
            await asyncio.sleep(2)  # Async 2 second delay between messages
            
            message_data = {
                'type': 'test_message',
                'count': i + 1,
                'message': f'ASYNC Test message {i + 1} - This should appear with 2 second delays',
                'timestamp': time.time()
            }
            
            yield f"data: {json.dumps(message_data)}\n\n"
            yield f": heartbeat {i + 1} {' ' * 512}\n\n"  # Padding after each message to force flush
            logger.debug(f"Sent async test message {i + 1}")
        
        # Send completion message
        await asyncio.sleep(1)
        yield f"data: {json.dumps({'type': 'completed', 'message': 'Async SSE test completed successfully', 'timestamp': time.time()})}\n\n"
        
        logger.info("Async simple SSE test stream completed")
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Headers'] = 'Cache-Control'
    return response


async def search_status_stream(request, search_id):
    """
    Enhanced async SSE endpoint with Redis pub/sub for real-time progress updates.
    Falls back to database polling for compatibility.
    """
    async def event_stream():
        import redis
        
        logger.info(f"Starting enhanced async SSE stream for search request {search_id}")
        
        # Try Redis pub/sub for real-time updates first
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=3, decode_responses=True)
            redis_client.ping()  # Test connection
            
            pubsub = redis_client.pubsub()
            channel = f'search.{search_id}.progress'
            
            try:
                pubsub.subscribe(channel)
                logger.info(f"🚀 Using Redis pub/sub for real-time updates: {channel}")
                
                # Send connection confirmation
                yield f"data: {json.dumps({
                    'type': 'connected',
                    'message': '🚀 Real-time progress tracking active',
                    'mode': 'redis_pubsub'
                })}\n\n"
                yield f": padding to prevent buffering {' ' * 1024}\n\n"  # Force flush with padding
                
                # Listen for Redis messages with proper real-time streaming
                start_time = time.time()
                max_duration = 300  # 5 minutes total timeout
                
                while True:
                    # Use get_message with timeout for proper real-time streaming
                    message = pubsub.get_message(timeout=1.0)
                    
                    if message is not None:
                        if message['type'] == 'message':
                            # Forward Redis message directly to SSE immediately
                            yield f"data: {message['data']}\n\n"
                            yield f": heartbeat {' ' * 256}\n\n"  # Padding after each message to force flush
                            
                            # Check for completion
                            try:
                                data = json.loads(message['data'])
                                if data.get('type') in ['complete', 'error']:
                                    logger.info(f"Stream completion detected: {data.get('type')}")
                                    break
                            except:
                                pass
                                
                        elif message['type'] == 'subscribe':
                            logger.debug("Successfully subscribed to Redis channel")
                    
                    # Small async yield to prevent blocking
                    await asyncio.sleep(0.1)
                    
                    # Check for overall timeout
                    if time.time() - start_time > max_duration:
                        yield f"data: {json.dumps({
                            'type': 'timeout',
                            'message': 'Stream timeout reached'
                        })}\n\n"
                        break
                        
            except Exception as redis_error:
                logger.warning(f"Redis pub/sub error, falling back to polling: {redis_error}")
                pubsub.close()
                raise redis_error
            finally:
                try:
                    pubsub.close()
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Redis unavailable, using database polling fallback: {e}")
            
            # Fallback to original database polling method (async version)
            yield f"data: {json.dumps({
                'type': 'connected',
                'message': '📊 Progress tracking via database polling',
                'mode': 'database_polling'
            })}\n\n"
            yield f": padding to prevent buffering {' ' * 1024}\n\n"  # Force flush with padding
            
            max_polls = 150  # Maximum number of polling attempts (5 minutes at 2s intervals)
            poll_count = 0
            
            while poll_count < max_polls:
                poll_count += 1
                
                # Get search request
                try:
                    search_request = SearchRequestModel.objects.select_related('search_session').get(
                        search_id=search_id
                    )
                except SearchRequestModel.DoesNotExist:
                    yield f"data: {json.dumps({
                        'type': 'error',
                        'message': 'Search request not found'
                    })}\n\n"
                    break
                
                # Get URL requests linked to this search
                url_requests = URLRequestTable.objects.filter(
                    search_request=search_request
                ).exclude(celery_task_id__isnull=True).select_related('video_metadata')
                
                if not url_requests.exists():
                    yield f"data: {json.dumps({
                        'type': 'waiting',
                        'message': 'Waiting for video processing to start...',
                        'poll_count': poll_count
                    })}\n\n"
                    await asyncio.sleep(2)  # Async sleep instead of time.sleep
                    continue
                
                total_progress = 0
                video_count = 0
                video_progress_data = []
                completed_count = 0
                
                for url_request in url_requests:
                    video_count += 1
                    current_progress = 0
                    description = 'Starting...'
                    status_val = url_request.status
                    
                    # Get video metadata for title
                    video_title = None
                    if hasattr(url_request, 'video_metadata') and url_request.video_metadata:
                        video_title = url_request.video_metadata.title
                    
                    # Determine progress based on status
                    if url_request.status in ['success', 'failed']:
                        current_progress = 100
                        completed_count += 1
                        description = 'Completed successfully' if url_request.status == 'success' else 'Processing failed'
                    elif url_request.celery_task_id:
                        # Get progress from celery for in-progress tasks
                        try:
                            async_result = AsyncResult(url_request.celery_task_id)
                            progress_obj = Progress(async_result)
                            progress_info = progress_obj.get_info()
                            
                            if progress_info:
                                current_progress = progress_info.get('current', 0)
                                description = progress_info.get('description', 'Processing...')
                                
                                # Check if task is actually complete
                                if async_result.state in ['SUCCESS', 'FAILURE']:
                                    current_progress = 100
                                    completed_count += 1
                                    if async_result.state == 'SUCCESS':
                                        description = 'Completed successfully'
                                    else:
                                        description = 'Processing failed'
                        except Exception as e:
                            logger.error(f"Error getting progress for task {url_request.celery_task_id}: {e}")
                            description = 'Error checking progress'
                    
                    total_progress += current_progress
                    video_progress_data.append({
                        'video_id': str(url_request.request_id),
                        'url': url_request.url,
                        'title': video_title,
                        'progress': current_progress,
                        'description': description,
                        'status': status_val
                    })
                
                # Calculate overall progress
                overall_progress = (total_progress / video_count) if video_count > 0 else 0
                
                # Send overall progress
                yield f"data: {json.dumps({
                    'type': 'progress_update',
                    'search_id': str(search_request.search_id),
                    'overall_progress': round(overall_progress, 2),
                    'completed_videos': completed_count,
                    'total_videos': video_count,
                    'videos': video_progress_data,
                    'query': search_request.original_query,
                    'poll_count': poll_count
                })}\n\n"
                yield f": heartbeat {poll_count} {' ' * 256}\n\n"  # Padding after each message to force flush
                
                # Check if all processing is complete
                if completed_count >= video_count and video_count > 0:
                    # For already-failed videos, give frontend time to process updates
                    # Only complete immediately after a few polls to avoid abrupt termination
                    if poll_count >= 3:
                        # Send final completion message
                        successful_videos = len([v for v in video_progress_data if v['status'] == 'success'])
                        failed_videos = len([v for v in video_progress_data if v['status'] == 'failed'])
                        
                        yield f"data: {json.dumps({
                            'type': 'processing_completed',
                            'search_id': str(search_request.search_id),
                            'total_videos': video_count,
                            'completed_videos': completed_count,
                            'successful_videos': successful_videos,
                            'failed_videos': failed_videos,
                            'message': 'All video processing completed'
                        })}\n\n"
                        break
                    
                await asyncio.sleep(2)  # Async poll every 2 seconds
            
            # If we hit max polls, send timeout message
            if poll_count >= max_polls:
                yield f"data: {json.dumps({
                    'type': 'timeout',
                    'message': 'Stream timeout reached',
                    'poll_count': poll_count
                })}\n\n"
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Headers'] = 'Cache-Control'
    response['Connection'] = 'keep-alive'
    response['Transfer-Encoding'] = 'chunked'  # Force chunked encoding
    return response


class IntegratedSearchProcessAPIView(APIView):
    """
    Integrated search and video processing endpoint with enhanced features.
    
    This endpoint combines search and parallel video processing into a single
    optimized workflow using the most efficient processing pipeline.
    """
    
    def post(self, request):
        """Start integrated search and video processing workflow"""
        try:
            # Simple validation for basic usage
            query = request.data.get('query', '').strip()
            max_videos = request.data.get('max_videos', 5)
            
            if not query:
                return Response(
                    {'error': 'Query parameter is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate max_videos
            try:
                max_videos = int(max_videos)
                if max_videos < 1 or max_videos > 20:
                    max_videos = 20
            except (ValueError, TypeError):
                max_videos = 5
            
            logger.info(f"Starting integrated workflow for query: '{query}'")
            
            # Create session  
            session = get_or_create_session(request)
            
            # Create search request
            with transaction.atomic():
                search_request = SearchRequestModel.objects.create(
                    search_session=session,
                    original_query=query,
                    status='processing'
                )
                
                logger.info(f"Created search request {search_request.search_id}")
                
            # Start integrated processing workflow after transaction commits
            task = process_search_with_videos.delay(
                str(search_request.search_id),
                max_videos=max_videos,
                start_video_processing=True
            )
            
            logger.info(f"Started integrated processing task {task.id}")
            
            # Prepare response
            response_data = {
                'search_id': str(search_request.search_id),
                'session_id': str(session.session_id),
                'query': query,
                'status': 'processing',
            }
            
            # Return response directly for now (skip serializer for simplicity)
            return Response(response_data, status=status.HTTP_202_ACCEPTED)
                
        except Exception as e:
            logger.error(f"Error in IntegratedSearchProcessAPIView: {e}")
            return Response(
                {'error': f'Internal server error: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
