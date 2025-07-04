from celery import shared_task
import yt_dlp
from django.db import transaction
import logging
from datetime import datetime

from api.models import URLRequestTable
from ..models import VideoMetadata
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..validators import validate_video_info
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)

def parse_upload_date(upload_date_str):
    """Parse YouTube upload_date string (YYYYMMDD) to date object"""
    if not upload_date_str:
        return None
    try:
        return datetime.strptime(upload_date_str, '%Y%m%d').date()
    except (ValueError, TypeError):
        logger.warning(f"Could not parse upload_date: {upload_date_str}")
        return None

def extract_thumbnail_url(info):
    """Extract the best thumbnail URL from thumbnails array"""
    thumbnails = info.get('thumbnails', [])
    if not thumbnails:
        return info.get('thumbnail', '')
    
    # Find the highest quality thumbnail
    best_thumbnail = max(thumbnails, key=lambda x: x.get('width', 0) * x.get('height', 0), default={})
    return best_thumbnail.get('url', '')

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
            video_id = validated_info.get('id')
            
            # Check if VideoMetadata with this video_id already exists
            existing_metadata = None
            if video_id:
                existing_metadata = VideoMetadata.objects.filter(video_id=video_id).first()
            
            if existing_metadata and existing_metadata.url_request != url_request:
                # Video already exists for different request - link this request to existing metadata
                logger.info(f"Video {video_id} already exists, linking request {url_request_id} to existing metadata")
                
                # Delete any incomplete metadata for this request
                VideoMetadata.objects.filter(url_request=url_request).delete()
                
                # Update the existing metadata to point to this request (if it's more recent)
                if url_request.created_at > existing_metadata.url_request.created_at:
                    existing_metadata.url_request = url_request
                    existing_metadata.save()
                
                metadata_obj = existing_metadata
            else:
                # Create or update metadata normally
                metadata_obj, created = VideoMetadata.objects.get_or_create(
                    url_request=url_request,
                    defaults={
                        # Existing fields
                        'video_id': video_id,
                        'title': validated_info.get('title', ''),
                        'description': validated_info.get('description', ''),
                        'duration': validated_info.get('duration'),
                        'channel_name': validated_info.get('uploader') or validated_info.get('channel', ''),
                        'view_count': validated_info.get('view_count'),
                        
                        # New fields from YouTube API
                        'upload_date': parse_upload_date(validated_info.get('upload_date')),
                        'language': validated_info.get('language', 'en'),
                        'like_count': validated_info.get('like_count'),
                        'channel_id': validated_info.get('channel_id', ''),
                        'tags': validated_info.get('tags', []),
                        'categories': validated_info.get('categories', []),
                        'thumbnail': extract_thumbnail_url(validated_info),
                        'channel_follower_count': validated_info.get('channel_follower_count'),
                        'channel_is_verified': validated_info.get('channel_is_verified', False),
                        'uploader_id': validated_info.get('uploader_id', ''),
                        
                        'status': 'processing'
                    }
                )
                
                # Update metadata if it already existed
                if not created:
                    # Existing fields
                    metadata_obj.video_id = video_id
                    metadata_obj.title = validated_info.get('title', '')
                    metadata_obj.description = validated_info.get('description', '')
                    metadata_obj.duration = validated_info.get('duration')
                    metadata_obj.channel_name = validated_info.get('uploader') or validated_info.get('channel', '')
                    metadata_obj.view_count = validated_info.get('view_count')
                    
                    # New fields from YouTube API
                    metadata_obj.upload_date = parse_upload_date(validated_info.get('upload_date'))
                    metadata_obj.language = validated_info.get('language', 'en')
                    metadata_obj.like_count = validated_info.get('like_count')
                    metadata_obj.channel_id = validated_info.get('channel_id', '')
                    metadata_obj.tags = validated_info.get('tags', [])
                    metadata_obj.categories = validated_info.get('categories', [])
                    metadata_obj.thumbnail = extract_thumbnail_url(validated_info)
                    metadata_obj.channel_follower_count = validated_info.get('channel_follower_count')
                    metadata_obj.channel_is_verified = validated_info.get('channel_is_verified', False)
                    metadata_obj.uploader_id = validated_info.get('uploader_id', '')
                    
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