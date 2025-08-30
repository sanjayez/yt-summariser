"""
Search Views - Handles video question answering and search endpoints.
Contains clean async views with proper RAG integration and error handling.
"""

import json
from uuid import UUID

from asgiref.sync import sync_to_async
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pydantic import ValidationError

from telemetry import get_logger, handle_exceptions, timed_operation

from ..models import URLRequestTable
from ..schemas import VideoQuestionRequest, VideoSearchResponse
from ..services.response_service import ResponseService
from ..services.search_service import SearchService

logger = get_logger(__name__)
search_service = SearchService()
response_service = ResponseService()


@csrf_exempt
@require_http_methods(["POST"])
@handle_exceptions(reraise=True)
@timed_operation()
async def ask_video_question(request: HttpRequest, request_id: UUID) -> JsonResponse:
    """
    Ask questions about a processed video using smart search strategy.

    This endpoint provides RAG-based question answering by:
    1. Validating the question using Pydantic schemas
    2. Retrieving video data and checking processing status
    3. Performing vector search when embeddings are available
    4. Falling back to transcript search when needed
    5. Returning formatted results with sources and confidence

    Args:
        request: HTTP request with JSON body containing 'question'
        request_id: UUID of the video processing request

    Returns:
        JsonResponse with answer, sources, confidence, and metadata
    """
    try:
        # Parse and validate request data
        try:
            data = json.loads(request.body)
            question_request = VideoQuestionRequest(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Invalid question request: {e}")
            error_response = response_service.format_error_response(
                error_type="Invalid request",
                message=f"Question validation failed: {str(e)}",
                status="bad_request",
            )
            return JsonResponse(error_response, status=400)

        # Get the request and related data with proper async handling
        try:
            url_request = await sync_to_async(
                URLRequestTable.objects.select_related(
                    "video_metadata__video_transcript"
                ).get
            )(request_id=request_id)
        except URLRequestTable.DoesNotExist:
            logger.warning(f"Video question request not found: {request_id}")
            error_response = response_service.format_error_response(
                error_type="Request not found",
                message=f"No video processing request found with ID: {request_id}",
                status="not_found",
                details={"request_id": str(request_id)},
            )
            return JsonResponse(error_response, status=404)

        # Check if video metadata exists
        if not hasattr(url_request, "video_metadata"):
            error_response = response_service.format_error_response(
                error_type="Video metadata not found",
                message="Video processing may not have started or failed during metadata extraction",
                status="not_found",
            )
            return JsonResponse(error_response, status=404)

        video_metadata = url_request.video_metadata

        # Check if transcript exists
        if not hasattr(video_metadata, "video_transcript"):
            error_response = response_service.format_error_response(
                error_type="Video transcript not found",
                message="Video processing may not have reached transcript extraction stage",
                status="not_found",
            )
            return JsonResponse(error_response, status=404)

        transcript = video_metadata.video_transcript

        # Check if video processing is complete
        if url_request.status == "processing":
            error_response = response_service.format_error_response(
                error_type="Video still processing",
                message="Please wait for video processing to complete before asking questions",
                status="processing",
            )
            return JsonResponse(error_response, status=202)

        elif url_request.status == "failed":
            error_response = response_service.format_error_response(
                error_type="Video processing failed",
                message="Cannot answer questions for failed video processing",
                status="failed",
            )
            return JsonResponse(error_response, status=500)

        # Check if transcript is available
        if not transcript.transcript_text or not transcript.transcript_text.strip():
            error_response = response_service.format_error_response(
                error_type="No transcript available",
                message="Cannot answer questions without video transcript",
                status="no_transcript",
            )
            return JsonResponse(error_response, status=404)

        # Perform search using appropriate method
        search_result = None
        question = question_request.question

        if video_metadata.is_embedded:
            try:
                logger.info(f"Using vector search for question: {question}")

                # Use the new search service with proper async patterns
                search_result = await search_service.search_video_content(
                    question, video_metadata, transcript
                )

            except Exception as e:
                logger.warning(
                    f"Vector search failed, falling back to transcript search: {e}"
                )
                search_result = None

        # Fallback to transcript search if vector search failed or not available
        if not search_result:
            logger.info(f"Using transcript fallback for question: {question}")
            search_result = await search_service.search_transcript_fallback(
                question, transcript.transcript_text, video_metadata
            )

        # Format video metadata for response
        video_metadata_response = {
            "video_id": video_metadata.video_id,
            "title": video_metadata.title,
            "duration_string": video_metadata.duration_string,
            "youtube_url": str(video_metadata.webpage_url),
        }

        # Prepare validated response
        response_data = VideoSearchResponse(
            question=question,
            answer=search_result["answer"],
            sources=search_result["sources"],
            confidence=search_result["confidence"],
            search_method=search_result["search_method"],
            results_count=search_result["results_count"],
            video_metadata=video_metadata_response,
            timing=search_result.get("timing"),
        )

        logger.info(
            f"Question answered for video {video_metadata.video_id}: {question[:50]}..."
        )

        # Use Pydantic's model_dump_json method to properly serialize HttpUrl objects
        return HttpResponse(
            response_data.model_dump_json(), content_type="application/json", status=200
        )

    except Exception as e:
        logger.error(f"Question answering failed for {request_id}: {e}")
        error_response = response_service.format_error_response(
            error_type="Internal server error", message=str(e), status="error"
        )
        return JsonResponse(error_response, status=500)
