"""
Video Views - Handles video processing and summary retrieval endpoints.
Contains clean, focused async views with proper error handling and validation.
"""
import json
from typing import Dict, Any
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from asgiref.sync import sync_to_async
from pydantic import ValidationError
from uuid import UUID

from telemetry import get_logger, handle_exceptions, timed_operation
from ..models import URLRequestTable
from ..schemas import VideoProcessRequest, VideoProcessResponse, VideoSummaryResponse
from ..services.response_service import ResponseService
from ..utils.get_client_ip import get_client_ip


logger = get_logger(__name__)
response_service = ResponseService()


@csrf_exempt
@require_http_methods(["POST"])
@handle_exceptions(reraise=True)
@timed_operation()
async def process_single_video(request: HttpRequest) -> JsonResponse:
    """
    Process a single YouTube video through the complete pipeline.
    
    This endpoint initiates video processing by:
    1. Validating the YouTube URL using Pydantic schemas
    2. Creating a URLRequestTable record
    3. Triggering the existing Celery workflow (no changes to video_processor)
    4. Returning the request ID for status tracking
    
    Args:
        request: HTTP request with JSON body containing 'url'
        
    Returns:
        JsonResponse with request_id, url, status, and message
    """
    try:
        # Parse and validate request data
        try:
            data = json.loads(request.body)
            video_request = VideoProcessRequest(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Invalid video process request: {e}")
            error_response = response_service.format_error_response(
                error_type="Invalid request",
                message=f"Request validation failed: {str(e)}",
                status="bad_request"
            )
            return JsonResponse(error_response, status=400)
        
        # Get client IP address
        ip_address = get_client_ip(request)
        
        # Create URL request record
        url_request = await sync_to_async(URLRequestTable.objects.create)(
            url=str(video_request.url),
            ip_address=ip_address,
            status='processing'
        )
        
        # Trigger existing Celery workflow (NO CHANGES to video_processor)
        from video_processor.processors.workflow import process_youtube_video
        await sync_to_async(process_youtube_video.delay)(url_request.id)
        
        # Prepare validated response
        response_data = VideoProcessResponse(
            request_id=str(url_request.request_id),
            url=video_request.url,
            status="processing",
            message="Video processing started. Use the request_id to check status and retrieve results."
        )
        
        logger.info(f"Video processing initiated for URL: {str(video_request.url)[:50]}...")
        return HttpResponse(response_data.model_dump_json(), content_type='application/json', status=201)
        
    except Exception as e:
        logger.error(f"Video processing initiation failed: {e}")
        error_response = response_service.format_error_response(
            error_type="Processing failed",
            message=f"Failed to initiate video processing: {str(e)}",
            status="internal_error"
        )
        return JsonResponse(error_response, status=500)


@require_http_methods(["GET"])
@handle_exceptions(reraise=True)
@timed_operation()
async def get_video_summary(request: HttpRequest, request_id: UUID) -> JsonResponse:
    """
    Get the AI-generated summary and key points for a processed video.
    
    This endpoint retrieves:
    1. Video summary and key points
    2. Video metadata
    3. Processing status and timestamps
    
    Args:
        request: HTTP request
        request_id: UUID of the video processing request
        
    Returns:
        JsonResponse with summary, key_points, video_metadata, and status
    """
    try:
        # Get the request and related data with proper async handling
        try:
            url_request = await sync_to_async(
                URLRequestTable.objects.select_related(
                    'video_metadata__video_transcript'
                ).get
            )(request_id=request_id)
        except URLRequestTable.DoesNotExist:
            logger.warning(f"Video summary request not found: {request_id}")
            error_response = response_service.format_error_response(
                error_type="Request not found",
                message=f"No video processing request found with ID: {request_id}",
                status="not_found",
                details={"request_id": str(request_id)}
            )
            return JsonResponse(error_response, status=404)
        
        # Check if video metadata exists
        if not hasattr(url_request, 'video_metadata'):
            error_response = response_service.format_error_response(
                error_type="Video metadata not found",
                message="Video processing may not have started or failed during metadata extraction",
                status="not_found"
            )
            return JsonResponse(error_response, status=404)
        
        video_metadata = url_request.video_metadata
        
        # Check if transcript exists
        if not hasattr(video_metadata, 'video_transcript'):
            error_response = response_service.format_error_response(
                error_type="Video transcript not found",
                message="Video processing may not have reached transcript extraction stage",
                status="not_found"
            )
            return JsonResponse(error_response, status=404)
        
        transcript = video_metadata.video_transcript
        
        # Check processing status
        if url_request.status == 'processing':
            processing_response = {
                'status': 'processing',
                'message': 'Video is still being processed. Please check back later.',
                'stages': {
                    'metadata_extracted': video_metadata.status == 'success',
                    'transcript_extracted': transcript.status == 'success',
                    'summary_generated': bool(transcript.summary and transcript.summary.strip())
                }
            }
            return JsonResponse(processing_response, status=202)
        
        elif url_request.status == 'failed':
            error_response = response_service.format_error_response(
                error_type="Video processing failed",
                message="Video processing encountered errors and could not be completed",
                status="failed"
            )
            return JsonResponse(error_response, status=500)
        
        # Check if summary is available
        if not transcript.summary or not transcript.summary.strip():
            error_response = response_service.format_error_response(
                error_type="Summary not available",
                message="Video was processed but summary generation failed or is not yet complete",
                status="no_summary"
            )
            return JsonResponse(error_response, status=404)
        
        # Format video metadata using response service
        video_metadata_formatted = response_service.format_video_metadata(video_metadata)
        
        # Prepare validated response
        response_data = VideoSummaryResponse(
            summary=transcript.summary,
            key_points=transcript.key_points if transcript.key_points else [],
            video_metadata=video_metadata_formatted,
            status="completed",
            generated_at=transcript.created_at.isoformat() if transcript.created_at else None,
            summary_length=len(transcript.summary),
            key_points_count=len(transcript.key_points) if transcript.key_points else 0
        )
        
        logger.info(f"Video summary retrieved for request: {request_id}")
        return HttpResponse(response_data.model_dump_json(), content_type='application/json', status=200)
        
    except Exception as e:
        logger.error(f"Video summary retrieval failed for {request_id}: {e}")
        error_response = response_service.format_error_response(
            error_type="Internal server error",
            message=str(e),
            status="error"
        )
        return JsonResponse(error_response, status=500)