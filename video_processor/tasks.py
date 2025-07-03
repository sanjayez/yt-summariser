from celery import shared_task, chain, group, chord
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from django.db import transaction
import logging

from api.models import URLRequestTable
from .models import VideoMetadata, VideoTranscript, update_url_request_status
from .config import YOUTUBE_CONFIG, TASK_STATES, validate_youtube_url, validate_video_info, validate_transcript_data
from .utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)

# Main entry point - creates task chain
@shared_task(bind=True)
def process_youtube_video(self, url_request_id):
    """
    Entry point that creates and executes the video processing chain.
    Uses Celery's chain primitive for proper task orchestration.
    """
    try:
        # Validate input
        url_request = URLRequestTable.objects.select_related('video_metadata', 'video_transcript').get(id=url_request_id)
        
        # Validate URL
        validate_youtube_url(url_request.url)
        
        logger.info(f"Starting video processing pipeline for request {url_request_id}")
        
        parallel_tasks = group(
            extract_video_transcript.s(url_request_id),
        )
        
        workflow = chain(
            extract_video_metadata.s(url_request_id),
            chord(parallel_tasks, update_overall_status.s(url_request_id))
        )
        
        # Execute workflow
        result = workflow.apply_async()
        
        return f"Initiated processing pipeline for request {url_request_id}"
        
    except Exception as e:
        logger.error(f"Failed to initiate processing pipeline for {url_request_id}: {e}")
        handle_dead_letter_task('process_youtube_video', self.request.id, [url_request_id], {}, e)
        raise

# Task 1: Extract Video Metadata
@shared_task(bind=True, 
             autoretry_for=(Exception,), 
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata']['jitter'],
             retry_kwargs=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata'])
@idempotent_task
def extract_video_metadata(self, url_request_id):
    """
    Atomic task: Extract video metadata using yt-dlp with timeout protection.
    """
    try:
        # Update progress
        update_task_progress(self, TASK_STATES['EXTRACTING_METADATA'], 10)
        
        # Optimized query with select_related
        url_request = URLRequestTable.objects.select_related('video_metadata').get(id=url_request_id)
        
        logger.info(f"Extracting metadata for request {url_request_id}")
        
        # Extract video info with timeout protection
        with timeout(YOUTUBE_CONFIG['TASK_TIMEOUTS']['metadata_timeout'], "Metadata extraction"):
            with yt_dlp.YoutubeDL(YOUTUBE_CONFIG['YDL_OPTS']) as ydl:
                info = ydl.extract_info(url_request.url, download=False)
        
        # Validate extracted info
        validated_info = validate_video_info(info)
        
        update_task_progress(self, TASK_STATES['EXTRACTING_METADATA'], 50)
        
        # Database operations with transaction
        with transaction.atomic():
            metadata_obj, created = VideoMetadata.objects.get_or_create(
                url_request=url_request,
                defaults={
                    'title': validated_info.get('title', ''),
                    'description': validated_info.get('description', ''),
                    'duration': validated_info.get('duration'),
                    'channel_name': validated_info.get('uploader', ''),
                    'view_count': validated_info.get('view_count'),
                    'status': 'processing'
                }
            )
            
            # Update metadata if it already existed
            if not created:
                metadata_obj.title = validated_info.get('title', '')
                metadata_obj.description = validated_info.get('description', '')
                metadata_obj.duration = validated_info.get('duration')
                metadata_obj.channel_name = validated_info.get('uploader', '')
                metadata_obj.view_count = validated_info.get('view_count')
                metadata_obj.status = 'processing'
            
            # Mark metadata as successful
            metadata_obj.status = 'success'
            metadata_obj.save()
        
        video_id = validated_info.get('id')
        update_task_progress(self, TASK_STATES['EXTRACTING_METADATA'], 100)
        
        logger.info(f"Successfully extracted metadata for: {validated_info.get('title', 'Unknown')}")
        
        # Return video_id for next task in chain
        return {'video_id': video_id, 'title': validated_info.get('title', 'Unknown')}
        
    except Exception as e:
        logger.error(f"Metadata extraction failed for request {url_request_id}: {str(e)}")
        
        # Handle failure with transaction
        with transaction.atomic():
            try:
                url_request = URLRequestTable.objects.get(id=url_request_id)
                VideoMetadata.objects.update_or_create(
                    url_request=url_request,
                    defaults={'status': 'failed'}
                )
            except Exception as db_error:
                logger.error(f"Failed to update metadata status: {db_error}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('extract_video_metadata', self.request.id, [url_request_id], {}, e)
        
        raise

# Task 2: Extract Video Transcript
@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['jitter'],
             retry_kwargs=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript'])
@idempotent_task
def extract_video_transcript(self, metadata_result, url_request_id):
    """
    Atomic task: Extract transcript with timeout protection and validation.
    """
    try:
        video_id = metadata_result.get('video_id') if isinstance(metadata_result, dict) else None
        
        if not video_id:
            raise ValueError("No video_id received from metadata extraction")
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 10)
        
        # Optimized query
        url_request = URLRequestTable.objects.select_related('video_transcript').get(id=url_request_id)
        
        logger.info(f"Extracting transcript for video {video_id}")
        
        # Create transcript object with transaction
        with transaction.atomic():
            transcript_obj, created = VideoTranscript.objects.get_or_create(
                url_request=url_request,
                defaults={
                    'transcript_text': '',
                    'status': 'processing'
                }
            )
            
            if not created:
                transcript_obj.status = 'processing'
                transcript_obj.save()
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 30)
        
        # Extract transcript with timeout protection
        with timeout(YOUTUBE_CONFIG['TASK_TIMEOUTS']['transcript_timeout'], "Transcript extraction"):
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Validate transcript data
        validated_transcript = validate_transcript_data(transcript_data)
        transcript_text = ' '.join([item['text'] for item in validated_transcript])
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 70)
        
        # Save with transaction
        with transaction.atomic():
            transcript_obj.transcript_text = transcript_text
            transcript_obj.transcript_data = validated_transcript
            transcript_obj.status = 'success'
            transcript_obj.save()
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 100)
        
        logger.info(f"Successfully extracted transcript with {len(validated_transcript)} segments")
        
        return {
            'transcript_segments': len(validated_transcript),
            'video_title': metadata_result.get('title', 'Unknown') if isinstance(metadata_result, dict) else 'Unknown'
        }
        
    except Exception as e:
        logger.warning(f"Transcript extraction failed for video {video_id}: {e}")
        
        # Mark transcript as failed but don't stop the chain
        with transaction.atomic():
            try:
                url_request = URLRequestTable.objects.get(id=url_request_id)
                VideoTranscript.objects.update_or_create(
                    url_request=url_request,
                    defaults={
                        'transcript_text': '',
                        'status': 'failed'
                    }
                )
            except Exception as db_error:
                logger.error(f"Failed to update transcript status: {db_error}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('extract_video_transcript', self.request.id, [url_request_id], {}, e)
        
        # Return empty result but don't break the chain (graceful degradation)
        return {'transcript_segments': 0, 'video_title': 'Unknown', 'error': str(e)}

# Task 3: Update Overall Status
@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'],
             retry_kwargs=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update'])
@idempotent_task
def update_overall_status(self, transcript_result, url_request_id):
    """
    Atomic task: Update overall status with timeout protection.
    """
    try:
        update_task_progress(self, TASK_STATES['UPDATING_STATUS'], 50)
        
        # Optimized query with select_related
        url_request = URLRequestTable.objects.select_related('video_metadata', 'video_transcript').get(id=url_request_id)
        
        logger.info(f"Updating overall status for request {url_request_id}")
        
        # Update status with timeout protection
        with timeout(YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_update_timeout'], "Status update"):
            with transaction.atomic():
                update_url_request_status(url_request)
        
        # Refresh to get updated status
        url_request.refresh_from_db()
        
        update_task_progress(self, TASK_STATES['COMPLETED'], 100, {
            'final_status': url_request.status,
            'has_metadata': hasattr(url_request, 'video_metadata'),
            'has_transcript': hasattr(url_request, 'video_transcript'),
        })
        
        logger.info(f"Final status for request {url_request_id}: {url_request.status}")
        
        result = {
            'status': url_request.status,
            'metadata_status': getattr(url_request.video_metadata, 'status', None) if hasattr(url_request, 'video_metadata') else None,
            'transcript_status': getattr(url_request.video_transcript, 'status', None) if hasattr(url_request, 'video_transcript') else None,
            'transcript_segments': transcript_result.get('transcript_segments', 0) if isinstance(transcript_result, dict) else 0
        }
        
        return f"Processing complete - {result}"
        
    except Exception as e:
        logger.error(f"Failed to update overall status for {url_request_id}: {e}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('update_overall_status', self.request.id, [url_request_id], {}, e)
        
        raise