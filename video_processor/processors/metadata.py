from celery import shared_task
import yt_dlp
from django.db import transaction
import logging

from api.models import URLRequestTable
from ..models import VideoMetadata
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..validators import validate_video_info
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)

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
                    'video_id': validated_info.get('id'),
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
                metadata_obj.video_id = validated_info.get('id')
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