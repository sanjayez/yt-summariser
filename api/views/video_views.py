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
        
        # TODO: Simplify LLM response parsing once responses are more consistent
        # TODO: Remove complex validation after fixing LLM prompt reliability
        logger.debug(f"Starting serialization for video {transcript.video_id}")
        
        # Handle summary data with comprehensive validation
        summary_text = ""
        try:
            if transcript.summary:
                raw_summary = transcript.summary
                logger.debug(f"Raw summary type: {type(raw_summary)}, length: {len(str(raw_summary)) if raw_summary else 0}")
                
                # Handle different summary data types
                if isinstance(raw_summary, bytes):
                    # Handle binary data
                    try:
                        summary_text = raw_summary.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        logger.error(f"Failed to decode binary summary for video {transcript.video_id}")
                        summary_text = ""
                elif isinstance(raw_summary, dict):
                    # Handle JSON object - extract text content
                    logger.warning(f"Summary is dict for video {transcript.video_id}, extracting text")
                    summary_text = str(raw_summary.get('text', raw_summary.get('summary', str(raw_summary)))).strip()
                elif isinstance(raw_summary, list):
                    # Handle array - join elements
                    logger.warning(f"Summary is list for video {transcript.video_id}, joining elements")
                    summary_text = ' '.join(str(item) for item in raw_summary if item).strip()
                else:
                    # Handle string or other types
                    summary_text = str(raw_summary).strip()
                
                # Additional validation for summary content
                if not summary_text:
                    logger.warning(f"Summary converted to empty string for video {transcript.video_id}")
                elif len(summary_text) < 10:
                    logger.warning(f"Summary too short for video {transcript.video_id}: {len(summary_text)} chars")
                
                # Check if summary contains error messages or invalid content
                error_indicators = [
                    "generation failed",
                    "processing encountered issues", 
                    "timed out",
                    "unavailable",
                    "failed to generate",
                    "error occurred",
                    "could not be processed",
                    "processing error",
                    "transcript unavailable",
                    "service temporarily unavailable"
                ]
                
                # Check for NULL/None strings that got converted
                null_indicators = ["null", "none", "undefined", "nan"]
                
                summary_lower = summary_text.lower()
                if any(indicator in summary_lower for indicator in error_indicators):
                    logger.warning(f"Summary contains error message for video {transcript.video_id}: {summary_text[:100]}...")
                    error_response = response_service.format_error_response(
                        error_type="Summary generation failed",
                        message="Video was processed but summary generation encountered errors",
                        status="summary_error",
                        details={"summary_error": summary_text}
                    )
                    return JsonResponse(error_response, status=422)
                elif summary_lower in null_indicators:
                    logger.warning(f"Summary contains null-like value for video {transcript.video_id}: {summary_text}")
                    summary_text = ""
                
            else:
                logger.warning(f"Summary is None/empty for video {transcript.video_id}")
                
        except Exception as e:
            logger.error(f"Error processing summary for video {transcript.video_id}: {e}")
            summary_text = ""
        
        # Final summary validation
        if not summary_text or len(summary_text.strip()) == 0:
            logger.warning(f"Final summary validation failed for video {transcript.video_id}")
            error_response = response_service.format_error_response(
                error_type="Summary not available",
                message="Video was processed but no valid summary was generated",
                status="no_summary"
            )
            return JsonResponse(error_response, status=404)
        
        # TODO: Consider data migration for old malformed summary data
        # Enhanced robust key_points processing with comprehensive type validation
        simple_key_points = []
        chapters_data = None
        
        try:
            # Handle different key_points data formats with extensive logging
            raw_key_points = transcript.key_points
            logger.debug(f"Raw key_points type: {type(raw_key_points)}, is_none: {raw_key_points is None}")
            
            # Log sample of raw data for debugging (truncated for security)
            if raw_key_points is not None:
                sample_data = str(raw_key_points)[:200] if len(str(raw_key_points)) > 200 else str(raw_key_points)
                logger.debug(f"Key points data sample for video {transcript.video_id}: {sample_data}")
            
            if raw_key_points is None:
                logger.info(f"key_points is None for video {transcript.video_id}")
                simple_key_points = []
                chapters_data = None
                
            elif isinstance(raw_key_points, bytes):
                # Handle binary data
                logger.warning(f"key_points is bytes for video {transcript.video_id}, attempting to decode")
                try:
                    decoded_data = raw_key_points.decode('utf-8')
                    try:
                        raw_key_points = json.loads(decoded_data)
                        logger.debug(f"Successfully decoded and parsed binary key_points for video {transcript.video_id}")
                    except json.JSONDecodeError:
                        # Treat as plain text
                        simple_key_points = [decoded_data.strip()] if decoded_data.strip() else []
                        chapters_data = None
                except UnicodeDecodeError:
                    logger.error(f"Failed to decode binary key_points for video {transcript.video_id}")
                    simple_key_points = []
                    chapters_data = None
                    
            elif isinstance(raw_key_points, str):
                # Handle string data - could be JSON or plain text
                logger.warning(f"key_points is string for video {transcript.video_id}, attempting to parse")
                
                # Check if it looks like JSON
                raw_key_points_stripped = raw_key_points.strip()
                if raw_key_points_stripped.startswith(('[', '{')):
                    try:
                        parsed_data = json.loads(raw_key_points_stripped)
                        raw_key_points = parsed_data
                        logger.debug(f"Successfully parsed JSON string key_points for video {transcript.video_id}")
                    except json.JSONDecodeError as je:
                        logger.error(f"Failed to parse JSON string key_points for video {transcript.video_id}: {je}")
                        # Treat as plain text key point
                        simple_key_points = [raw_key_points_stripped] if raw_key_points_stripped else []
                        chapters_data = None
                else:
                    # Treat as plain text key point
                    logger.debug(f"Treating string key_points as plain text for video {transcript.video_id}")
                    simple_key_points = [raw_key_points_stripped] if raw_key_points_stripped else []
                    chapters_data = None
                    
            elif isinstance(raw_key_points, dict):
                # Handle dictionary - might be a single chapter or wrapped data
                logger.warning(f"key_points is dict for video {transcript.video_id}, attempting to process")
                
                if 'chapters' in raw_key_points and isinstance(raw_key_points['chapters'], list):
                    # Wrapped chapters data
                    raw_key_points = raw_key_points['chapters']
                    logger.debug(f"Extracted chapters from wrapper dict for video {transcript.video_id}")
                elif 'key_points' in raw_key_points and isinstance(raw_key_points['key_points'], list):
                    # Wrapped key points data
                    raw_key_points = raw_key_points['key_points']
                    logger.debug(f"Extracted key_points from wrapper dict for video {transcript.video_id}")
                else:
                    # Treat the dict as a single chapter
                    raw_key_points = [raw_key_points]
                    logger.debug(f"Treating dict as single chapter for video {transcript.video_id}")
                    
            # Process the data if it's now a list (from string parsing or dict extraction)
            if isinstance(raw_key_points, list):
                if not raw_key_points:
                    # Empty list
                    logger.info(f"Empty key_points list for video {transcript.video_id}")
                    simple_key_points = []
                    chapters_data = None
                    
                else:
                    # Analyze list contents with more detailed checking
                    dict_count = sum(1 for item in raw_key_points if isinstance(item, dict))
                    string_count = sum(1 for item in raw_key_points if isinstance(item, str))
                    other_count = len(raw_key_points) - dict_count - string_count
                    
                    logger.debug(f"Key points analysis for video {transcript.video_id}: "
                               f"dicts={dict_count}, strings={string_count}, other={other_count}, total={len(raw_key_points)}")
                    
                    # Determine processing strategy based on content analysis
                    if dict_count > 0 and dict_count >= string_count:
                        # Structured chapters format (majority are dicts)
                        logger.info(f"Processing structured chapters for video {transcript.video_id}: {dict_count} chapter objects")
                        
                        # Validate chapter structure and extract simple key points
                        validated_chapters = []
                        for i, chapter in enumerate(raw_key_points):
                            if not isinstance(chapter, dict):
                                logger.warning(f"Skipping non-dict item at index {i} for video {transcript.video_id}: {type(chapter)}")
                                continue
                                
                            # Create validated chapter with comprehensive fallbacks
                            try:
                                validated_chapter = {
                                    "chapter": chapter.get("chapter", chapter.get("chapter_number", i + 1)),
                                    "title": str(chapter.get("title", chapter.get("heading", chapter.get("name", f"Chapter {i + 1}")))).strip(),
                                    "summary": str(chapter.get("summary", chapter.get("description", chapter.get("content", "")))).strip()
                                }
                                
                                # Validate and clean title
                                if not validated_chapter["title"] or validated_chapter["title"].lower() in ["null", "none", "undefined"]:
                                    validated_chapter["title"] = f"Chapter {i + 1}"
                                elif len(validated_chapter["title"]) > 300:
                                    validated_chapter["title"] = validated_chapter["title"][:297] + "..."
                                
                                # Validate and clean summary
                                if not validated_chapter["summary"] or validated_chapter["summary"].lower() in ["null", "none", "undefined"]:
                                    validated_chapter["summary"] = "No summary available"
                                elif len(validated_chapter["summary"]) > 2000:
                                    validated_chapter["summary"] = validated_chapter["summary"][:1997] + "..."
                                
                                # Validate chapter number
                                try:
                                    chapter_num = int(validated_chapter["chapter"])
                                    validated_chapter["chapter"] = max(1, chapter_num)  # Ensure positive
                                except (ValueError, TypeError):
                                    validated_chapter["chapter"] = i + 1
                                
                                validated_chapters.append(validated_chapter)
                                
                                # Extract title for simple key points
                                title = validated_chapter["title"]
                                if len(title) > 150:
                                    title = title[:147] + "..."
                                simple_key_points.append(title)
                                
                            except Exception as ce:
                                logger.warning(f"Error processing chapter {i} for video {transcript.video_id}: {ce}")
                                continue
                        
                        chapters_data = validated_chapters if validated_chapters else None
                        logger.debug(f"Processed {len(validated_chapters)} structured chapters")
                        
                    else:
                        # Simple string list format or mixed content
                        logger.info(f"Processing as simple key points for video {transcript.video_id}: {len(raw_key_points)} items")
                        
                        # Clean and validate all entries, converting to strings
                        for i, point in enumerate(raw_key_points):
                            try:
                                if point is None:
                                    continue
                                
                                # Convert to string and clean
                                if isinstance(point, dict):
                                    # Extract meaningful text from dict
                                    str_point = str(point.get('title', point.get('text', point.get('content', str(point))))).strip()
                                else:
                                    str_point = str(point).strip()
                                
                                # Skip empty, null-like, or very short entries
                                if (str_point and 
                                    str_point.lower() not in ["null", "none", "undefined", "nan", "{}"] and
                                    len(str_point) > 2):
                                    
                                    # Truncate if too long
                                    if len(str_point) > 300:
                                        str_point = str_point[:297] + "..."
                                    
                                    simple_key_points.append(str_point)
                                    
                            except Exception as pe:
                                logger.warning(f"Error processing key point {i} for video {transcript.video_id}: {pe}")
                                continue
                        
                        chapters_data = None
                        logger.debug(f"Processed {len(simple_key_points)} simple key points")
                        
            else:
                # Unexpected data type after all processing attempts
                logger.error(f"Unexpected key_points type after processing for video {transcript.video_id}: {type(raw_key_points)}")
                
                # Final fallback - try to extract something useful
                try:
                    fallback_text = str(raw_key_points).strip()
                    if fallback_text and fallback_text.lower() not in ["null", "none", "undefined"]:
                        if len(fallback_text) > 300:
                            fallback_text = fallback_text[:297] + "..."
                        simple_key_points = [fallback_text]
                    else:
                        simple_key_points = []
                except:
                    simple_key_points = []
                
                chapters_data = None
                
        except Exception as e:
            logger.error(f"Critical error processing key_points for video {transcript.video_id}: {e}", exc_info=True)
            # Fallback to empty arrays
            simple_key_points = []
            chapters_data = None

        # Log comprehensive final serialization state
        logger.info(f"Serialization complete for video {transcript.video_id}: "
                   f"summary_length={len(summary_text)}, "
                   f"simple_key_points={len(simple_key_points)}, "
                   f"chapters={'present' if chapters_data else 'none'}")
        
        # Additional validation before response creation
        if chapters_data:
            logger.debug(f"Chapters data preview for video {transcript.video_id}: "
                        f"{len(chapters_data)} chapters, "
                        f"first_title='{chapters_data[0].get('title', 'N/A')[:50]}...' if chapters_data else 'N/A'")
        
        if simple_key_points:
            logger.debug(f"Key points preview for video {transcript.video_id}: "
                        f"first_point='{simple_key_points[0][:50]}...' if simple_key_points else 'N/A'")

        # Final data integrity checks before response preparation
        try:
            # Validate summary is still good
            if not summary_text or len(summary_text.strip()) < 5:
                logger.error(f"Summary validation failed at final stage for video {transcript.video_id}")
                error_response = response_service.format_error_response(
                    error_type="Invalid summary data",
                    message="Summary data became invalid during processing",
                    status="data_corruption"
                )
                return JsonResponse(error_response, status=500)
            
            # Validate key points integrity
            if simple_key_points:
                # Check for any None or invalid entries that slipped through
                valid_key_points = []
                for point in simple_key_points:
                    if point and isinstance(point, str) and len(point.strip()) > 0:
                        valid_key_points.append(point.strip())
                
                if len(valid_key_points) != len(simple_key_points):
                    logger.warning(f"Cleaned {len(simple_key_points) - len(valid_key_points)} invalid key points for video {transcript.video_id}")
                    simple_key_points = valid_key_points
            
            # Validate chapters integrity
            if chapters_data:
                valid_chapters = []
                for chapter in chapters_data:
                    if (isinstance(chapter, dict) and 
                        chapter.get('title') and 
                        isinstance(chapter.get('title'), str) and
                        len(chapter.get('title', '').strip()) > 0):
                        valid_chapters.append(chapter)
                
                if len(valid_chapters) != len(chapters_data):
                    logger.warning(f"Cleaned {len(chapters_data) - len(valid_chapters)} invalid chapters for video {transcript.video_id}")
                    chapters_data = valid_chapters if valid_chapters else None
            
            # Prepare validated response with comprehensive error handling
            logger.debug(f"Creating VideoSummaryResponse for video {transcript.video_id}")
            
            response_data = VideoSummaryResponse(
                summary=summary_text,
                key_points=simple_key_points,
                chapters=chapters_data,
                video_metadata=video_metadata_formatted,
                status="completed",
                generated_at=transcript.created_at.isoformat() if transcript.created_at else None,
                summary_length=len(summary_text),
                key_points_count=len(simple_key_points)
            )
            
            # Validate the response can be serialized to JSON
            try:
                json_response = response_data.model_dump_json()
                logger.debug(f"Response JSON created successfully for video {transcript.video_id}")
                
            except Exception as json_error:
                logger.error(f"JSON serialization test failed for video {transcript.video_id}: {json_error}")
                error_response = response_service.format_error_response(
                    error_type="JSON serialization failed",
                    message="Response data could not be serialized to JSON",
                    status="serialization_error"
                )
                return JsonResponse(error_response, status=500)
                
        except ValidationError as ve:
            logger.error(f"Pydantic validation failed for video {transcript.video_id}: {ve}")
            
            # Try to provide more specific error information
            error_details = {"validation_errors": []}
            if hasattr(ve, 'errors'):
                for error in ve.errors():
                    error_details["validation_errors"].append({
                        "field": ".".join(str(loc) for loc in error.get('loc', [])),
                        "message": error.get('msg', 'Unknown validation error'),
                        "type": error.get('type', 'unknown')
                    })
            
            error_response = response_service.format_error_response(
                error_type="Response validation failed",
                message=f"Failed to validate response data: {str(ve)}",
                status="validation_error",
                details=error_details
            )
            return JsonResponse(error_response, status=500)
        
        except Exception as e:
            logger.error(f"Unexpected error during response preparation for video {transcript.video_id}: {e}", exc_info=True)
            error_response = response_service.format_error_response(
                error_type="Response preparation failed",
                message=f"Unexpected error during response preparation: {str(e)}",
                status="preparation_error"
            )
            return JsonResponse(error_response, status=500)
        
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