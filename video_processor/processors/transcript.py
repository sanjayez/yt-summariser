from celery import shared_task
from youtube_transcript_api import YouTubeTranscriptApi
from django.db import transaction
import logging

from api.models import URLRequestTable
from ..models import VideoTranscript
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..validators import validate_transcript_data
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)

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