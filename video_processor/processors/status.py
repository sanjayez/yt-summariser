from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction
import logging
from celery_progress.backend import ProgressRecorder

from api.models import URLRequestTable
from ..models import update_url_request_status
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)


def _gather_complete_video_data(url_request_id: str) -> dict:
    """
    Gather complete video data for SSE completion event.
    
    Args:
        url_request_id: UUID of the URLRequestTable
        
    Returns:
        dict: Complete video data with metadata, transcript, and content analysis
    """
    try:
        # Use select_related to minimize database queries
        url_request = URLRequestTable.objects.select_related(
            'video_metadata',
            'video_metadata__video_transcript',
            'video_metadata__video_transcript__content_analysis'
        ).get(request_id=url_request_id)
        
        video_data = {}
        
        # Gather complete metadata
        if hasattr(url_request, 'video_metadata') and url_request.video_metadata:
            video_metadata = url_request.video_metadata
            video_data['metadata'] = {
                'video_id': video_metadata.video_id,
                'title': video_metadata.title,
                'description': video_metadata.description,
                'duration': video_metadata.duration,
                'channel_name': video_metadata.channel_name,
                'channel_id': video_metadata.channel_id,
                'channel_thumbnail': video_metadata.channel_thumbnail,
                'view_count': video_metadata.view_count,
                'like_count': video_metadata.like_count,
                'comment_count': video_metadata.comment_count,
                'upload_date': str(video_metadata.upload_date) if video_metadata.upload_date else None,
                'language': video_metadata.language,
                'thumbnail': video_metadata.thumbnail,
                'tags': video_metadata.tags,
                'categories': video_metadata.categories,
                'channel_follower_count': video_metadata.channel_follower_count,
                'channel_is_verified': video_metadata.channel_is_verified,
                'uploader_id': video_metadata.uploader_id,
                'engagement': video_metadata.engagement,  # This is a list of high engagement segments
                'is_embedded': video_metadata.is_embedded,
                'status': video_metadata.status
            }
            
            # Gather complete transcript data
            if hasattr(video_metadata, 'video_transcript') and video_metadata.video_transcript:
                transcript = video_metadata.video_transcript
                video_data['transcript'] = {
                    'transcript_text': transcript.transcript_text,
                    'summary': transcript.summary,
                    'key_points': transcript.key_points,
                    'chapters': transcript.chapters,
                    'transcript_source': transcript.transcript_source,
                    'status': transcript.status
                }
                
                # Gather complete content analysis
                if hasattr(transcript, 'content_analysis') and transcript.content_analysis:
                    analysis = transcript.content_analysis
                    video_data['content_analysis'] = {
                        'preliminary_analysis_status': analysis.preliminary_analysis_status,
                        'timestamped_analysis_status': analysis.timestamped_analysis_status,
                        'speaker_tones': analysis.speaker_tones,
                        'raw_ad_segments': analysis.raw_ad_segments,
                        'raw_filler_segments': analysis.raw_filler_segments,
                        'ad_segments': analysis.ad_segments,
                        'filler_segments': analysis.filler_segments,
                        'ad_duration_ratio': float(analysis.ad_duration_ratio) if analysis.ad_duration_ratio is not None else None,
                        'filler_duration_ratio': float(analysis.filler_duration_ratio) if analysis.filler_duration_ratio is not None else None,
                        'content_rating': float(analysis.content_rating) if analysis.content_rating is not None else None,
                        'is_preliminary_complete': analysis.is_preliminary_complete,
                        'is_final_complete': analysis.is_final_complete,
                        'is_complete': analysis.is_complete
                    }
        
        return video_data
        
    except Exception as e:
        logger.error(f"Error gathering complete video data for {url_request_id}: {e}", exc_info=True)
        return {}


# Task 3: Update Overall Status
@shared_task(bind=True,
             name='video_processor.update_overall_status',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_hard_limit'],
             max_retries=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['max_retries'],
             default_retry_delay=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['countdown'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'])
@idempotent_task
def update_overall_status(self, embedding_result, url_request_id):
    """
    Update overall processing status after workflow completion.
    
    Args:
        embedding_result (dict): Result from previous embedding task
        url_request_id (str): UUID of the URLRequestTable to process
        
    Returns:
        str: Status update result message
        
    Raises:
        Exception: If status update fails after retries
    """
    url_request = None
    
    # Check if video was excluded in previous stage
    if embedding_result and embedding_result.get('excluded'):
        logger.info(f"Video was excluded - ensuring final status is correctly set: {embedding_result.get('exclusion_reason')}")
        # Get URLRequest and ensure it's marked as excluded (not processing)
        url_request = URLRequestTable.objects.get(request_id=url_request_id)
        if url_request.status == 'processing':
            url_request.status = 'failed'
            url_request.failure_reason = 'excluded'
            url_request.save()
            logger.info(f"Updated excluded video status from processing to failed/excluded")
        
        return f"Video excluded in previous stage: {embedding_result.get('exclusion_reason')}"
    
    try:
        # Set initial progress using celery_progress
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(0, 100, "Updating status")
        
        update_task_progress(self, TASK_STATES['UPDATING_STATUS'], 50)
        
        # Optimized query with select_related to avoid N+1 queries
        url_request = URLRequestTable.objects.select_related(
            'video_metadata',
            'video_metadata__video_transcript'
        ).get(request_id=url_request_id)
        
        logger.info(f"Updating overall status for request {url_request_id}")
        
        # Check for transcript extraction failures and provide descriptive error messages
        transcript_failed = False
        failure_reason = None
        
        # Check if transcript exists and failed
        if hasattr(url_request, 'video_metadata') and url_request.video_metadata:
            try:
                video_transcript = url_request.video_metadata.video_transcript
                if video_transcript and video_transcript.status == 'failed':
                    transcript_failed = True
                    failure_reason = "Transcript extraction failed"
                    # Check embedding result for specific error details
                    if isinstance(embedding_result, dict) and embedding_result.get('error'):
                        error_msg = embedding_result.get('error', '')
                        if '429' in error_msg or 'Too Many Requests' in error_msg:
                            failure_reason = "Rate limited by YouTube - transcript temporarily unavailable"
                        elif 'No transcript' in error_msg:
                            failure_reason = "No transcript available for this video"
                        else:
                            failure_reason = "Transcript extraction failed"
            except:
                # If no transcript object exists, check embedding result
                if isinstance(embedding_result, dict) and embedding_result.get('transcript_chunks_embedded', 0) == 0:
                    if embedding_result.get('error'):
                        error_msg = embedding_result.get('error', '')
                        if '429' in error_msg or 'Too Many Requests' in error_msg:
                            transcript_failed = True
                            failure_reason = "Rate limited by YouTube - transcript temporarily unavailable"
        
        # Update status (this should be very fast)
        with transaction.atomic():
            # If transcript failed, provide clear error message to video metadata
            if transcript_failed and hasattr(url_request, 'video_metadata'):
                video_metadata = url_request.video_metadata
                if video_metadata.status == 'processing':
                    video_metadata.status = 'failed'
                    video_metadata.error_message = failure_reason or "Processing failed"
                    video_metadata.save()
                    logger.info(f"Marked video metadata as failed: {failure_reason}")
                
                # Set failure reason on URLRequest if transcript failed
                if not url_request.failure_reason:
                    url_request.failure_reason = 'no_transcript'
                    url_request.save()
            
            update_url_request_status(url_request)
        
        # Refresh to get updated status
        url_request.refresh_from_db()
        
        # Check if this video is part of a topic search
        is_part_of_search = False
        try:
            is_part_of_search = url_request.search_request is not None
        except AttributeError:
            # search_request field might not exist in older installations
            pass
        
        # Only gather complete video data for standalone video processing
        video_data = None
        if not is_part_of_search:
            logger.info(f"Gathering complete video data for standalone video {url_request_id}")
            video_data = _gather_complete_video_data(url_request_id)
        else:
            logger.info(f"Skipping video data gathering for search-related video {url_request_id}")
        
        update_task_progress(self, TASK_STATES['COMPLETED'], 100, {
            'final_status': url_request.status,
            'has_metadata': hasattr(url_request, 'video_metadata'),
            'has_transcript': hasattr(url_request, 'video_metadata') and hasattr(url_request.video_metadata, 'video_transcript'),
            'video_data': video_data  # Will be None for topic search videos
        })
        
        logger.info(f"Final status for request {url_request_id}: {url_request.status}")
        
        result = {
            'status': url_request.status,
            'metadata_status': getattr(url_request.video_metadata, 'status', None) if hasattr(url_request, 'video_metadata') else None,
            'transcript_status': getattr(url_request.video_metadata.video_transcript, 'status', None) if hasattr(url_request, 'video_metadata') and hasattr(url_request.video_metadata, 'video_transcript') else None,
            'transcript_segments': embedding_result.get('transcript_chunks_embedded', 0) if isinstance(embedding_result, dict) else 0,
            'failure_reason': failure_reason if transcript_failed else None,
            'error_message': getattr(url_request.video_metadata, 'error_message', None) if hasattr(url_request, 'video_metadata') else None
        }
        
        # Set final progress using celery_progress
        progress_recorder.set_progress(100, 100, "Status updated")
        
        return f"Processing complete - {result}"
        
    except SoftTimeLimitExceeded:
        # Status update task is approaching timeout - this should never happen as it should be very fast
        logger.error(f"Status update soft timeout reached for request {url_request_id} - this indicates a serious database issue")
        
        try:
            # Last-ditch effort to mark as failed
            if url_request:
                url_request.status = 'failed'
                url_request.save()
                logger.error(f"Emergency status update to failed for {url_request_id}")
                
        except Exception as cleanup_error:
            logger.critical(f"Failed to emergency update status during timeout - database may be down: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Status update critical timeout for request {url_request_id}")
        
    except Exception as e:
        logger.error(f"Failed to update overall status for {url_request_id}: {e}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('update_overall_status', self.request.id, [url_request_id], {}, e)
        
        raise 