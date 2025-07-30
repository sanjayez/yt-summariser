"""
Status Views - Handles real-time video processing status streaming.
Contains clean async views for Server-Sent Events (SSE) status updates.
"""
import json
import time
from typing import Generator, Dict, Any
from django.http import StreamingHttpResponse, HttpRequest
from django.views.decorators.http import require_http_methods
# No async operations needed in status_views, removing unused import
from uuid import UUID

from telemetry import get_logger, handle_exceptions
from ..models import URLRequestTable
from ..services.response_service import ResponseService
from video_processor.config import API_CONFIG


logger = get_logger(__name__)
response_service = ResponseService()


@require_http_methods(["GET"])
@handle_exceptions(reraise=True)
def video_status_stream(request: HttpRequest, request_id: UUID) -> StreamingHttpResponse:
    """
    Stream real-time status updates for video processing.
    
    This endpoint provides Server-Sent Events (SSE) streaming for:
    1. Overall processing status
    2. Individual stage completion status
    3. Progress percentage calculation
    4. Detailed stage information
    
    The streaming includes all processing stages:
    - Metadata extraction
    - Transcript extraction  
    - Summary generation
    - Content embedding
    
    Args:
        request: HTTP request
        request_id: UUID of the video processing request
        
    Returns:
        StreamingHttpResponse with text/event-stream content
    """
    
    def event_stream() -> Generator[str, None, None]:
        """
        Generate Server-Sent Events for video processing status.
        
        This function polls the database at configured intervals and yields
        formatted SSE data until processing is complete or fails.
        
        Yields:
            str: Formatted SSE data strings
        """
        max_attempts = API_CONFIG['POLLING']['status_check_max_attempts']
        poll_interval = API_CONFIG['POLLING']['status_check_interval']
        attempts = 0
        
        logger.info(f"Starting status stream for request: {request_id}")
        
        # Maximum time based on config (max_attempts * poll_interval seconds)
        while attempts < max_attempts:
            try:
                # Get URL request with related data to avoid N+1 queries
                url_request = URLRequestTable.objects.select_related(
                    'video_metadata',
                    'video_metadata__video_transcript'
                ).get(request_id=request_id)
                
                # Build enhanced status data
                status_data = _build_status_data(url_request)
                
                # Send data as SSE
                yield f"data: {json.dumps(status_data)}\\n\\n"
                
                # Stop streaming if processing is complete
                if url_request.status in ['success', 'failed']:
                    logger.info(f"Status streaming completed for request: {request_id}")
                    break
                    
                time.sleep(poll_interval)  # Wait based on config
                attempts += 1
                
            except URLRequestTable.DoesNotExist:
                error_data = response_service.format_error_response(
                    error_type="Request not found",
                    message=f"No video processing request found with ID: {request_id}",
                    status="not_found"
                )
                yield f"data: {json.dumps(error_data)}\\n\\n"
                logger.warning(f"Status stream request not found: {request_id}")
                break
                
            except Exception as e:
                error_data = response_service.format_error_response(
                    error_type="Streaming error",
                    message=str(e),
                    status="error"
                )
                yield f"data: {json.dumps(error_data)}\\n\\n"
                logger.error(f"Status streaming error for {request_id}: {e}")
                break
        
        # If we've exceeded max attempts, send timeout message
        if attempts >= max_attempts:
            timeout_data = response_service.format_error_response(
                error_type="Streaming timeout",
                message="Status streaming timed out. Processing may still be ongoing.",
                status="timeout"
            )
            yield f"data: {json.dumps(timeout_data)}\\n\\n"
            logger.warning(f"Status stream timeout for request: {request_id}")
    
    # Create SSE response with proper headers
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    response['Access-Control-Allow-Origin'] = '*'  # Allow CORS for SSE
    response['Access-Control-Allow-Headers'] = 'Cache-Control'
    
    return response


def _build_status_data(url_request) -> Dict[str, Any]:
    """
    Build comprehensive status data for the streaming response.
    
    This function extracts and formats all relevant status information
    from the URLRequestTable and related models.
    
    Args:
        url_request: URLRequestTable instance with related data
        
    Returns:
        Dictionary with comprehensive status information
    """
    # Initialize status data structure
    data = {
        'overall_status': url_request.status,
        'timestamp': time.time(),
        'metadata_status': None,
        'transcript_status': None,
        'summary_status': None,
        'embedding_status': None,
        'stages': {
            'metadata_extracted': False,
            'transcript_extracted': False,
            'summary_generated': False,
            'content_embedded': False,
            'processing_complete': False
        }
    }
    
    # Add metadata details if exists
    if hasattr(url_request, 'video_metadata'):
        metadata = url_request.video_metadata
        data['metadata_status'] = metadata.status
        data['stages']['metadata_extracted'] = metadata.status == 'success'
        
        # Add embedding status
        if hasattr(metadata, 'is_embedded'):
            data['embedding_status'] = 'success' if metadata.is_embedded else 'pending'
            data['stages']['content_embedded'] = metadata.is_embedded
    
    # Add transcript details if exists through VideoMetadata
    if (hasattr(url_request, 'video_metadata') and 
        hasattr(url_request.video_metadata, 'video_transcript')):
        transcript = url_request.video_metadata.video_transcript
        data['transcript_status'] = transcript.status
        data['stages']['transcript_extracted'] = transcript.status == 'success'
        
        # Add summary status
        if transcript.summary:
            data['summary_status'] = 'success'
            data['stages']['summary_generated'] = True
        else:
            data['summary_status'] = 'pending' if transcript.status == 'success' else 'waiting'
    
    # Overall completion status
    data['stages']['processing_complete'] = url_request.status in ['success', 'failed']
    
    # Add progress percentage
    completed_stages = sum(1 for stage in data['stages'].values() if stage)
    total_stages = len(data['stages'])
    data['progress_percentage'] = int((completed_stages / total_stages) * 100)
    
    # Add failure information if applicable
    if url_request.status == 'failed' and url_request.failure_reason:
        data['failure_reason'] = url_request.failure_reason
    
    # Add task tracking information
    if url_request.celery_task_id:
        data['celery_task_id'] = url_request.celery_task_id
    if url_request.chain_task_id:
        data['chain_task_id'] = url_request.chain_task_id
    
    return data