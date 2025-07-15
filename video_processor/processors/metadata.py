import yt_dlp
import logging
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.db import transaction
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

def fallback_yt_dlp_extraction(url):
    """Fallback metadata extraction using yt-dlp"""
    logger.info(f"Using yt-dlp fallback for metadata extraction: {url}")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    
    return info

# Task 1: Extract Video Metadata  
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def extract_video_metadata(self, url_request_id):
    """
    Extract metadata for a YouTube video using yt-dlp.
    
    Args:
        url_request_id (int): ID of the URLRequestTable to process
        
    Returns:
        dict: Video metadata with video_id and title
        
    Raises:
        Exception: If metadata extraction fails after retries
    """
    try:
        # Get the URL request
        url_request = URLRequestTable.objects.get(id=url_request_id)
        
        logger.info(f"Starting metadata extraction for request {url_request_id}")
        logger.info(f"Extracting metadata for {url_request.url}")
        
        # Extract metadata using yt-dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_request.url, download=False)
        
        logger.info(f"Successfully extracted metadata for {info.get('title', 'Unknown')}")
        
        # Parse the metadata
        metadata = {
            'id': info.get('id'),  # For validation
            'title': info.get('title', '').strip(),
            'description': info.get('description', ''),
            'duration': info.get('duration'),
            'channel_name': info.get('uploader', ''),
            'view_count': info.get('view_count'),
            'upload_date': parse_upload_date(info.get('upload_date')),
            'language': info.get('language', 'en'),
            'like_count': info.get('like_count'),
            'channel_id': info.get('uploader_id', ''),
            'tags': info.get('tags', []) or [],
            'categories': info.get('categories', []) or [],
            'thumbnail': info.get('thumbnail', ''),
            'channel_follower_count': info.get('uploader_subscriber_count'),
            'channel_is_verified': info.get('uploader_verified', False),
            'uploader_id': info.get('uploader_id', ''),
        }
        
        # Validate metadata
        validate_video_info(metadata)
        
        # Save to database with transaction
        with transaction.atomic():
            video_metadata = VideoMetadata.objects.create(
                video_id=metadata['id'],  # Convert id to video_id for database
                title=metadata['title'][:255],  # Ensure title fits in field
                description=metadata['description'],
                duration=metadata['duration'],
                channel_name=metadata['channel_name'][:100],  # Ensure channel name fits
                view_count=metadata['view_count'],
                upload_date=metadata['upload_date'],
                language=metadata['language'][:10],  # Ensure language fits
                like_count=metadata['like_count'],
                channel_id=metadata['channel_id'][:100],  # Ensure channel ID fits
                tags=metadata['tags'],
                categories=metadata['categories'],
                thumbnail=metadata['thumbnail'],
                channel_follower_count=metadata['channel_follower_count'],
                channel_is_verified=metadata['channel_is_verified'],
                uploader_id=metadata['uploader_id'][:100],  # Ensure uploader ID fits
                url_request=url_request,
                status=TASK_STATES['COMPLETED']
            )
        
        logger.info(f"Metadata extraction completed successfully for video {metadata['id']}")
        
        # Return the expected format for workflow chain
        return {
            'video_id': metadata['id'], 
            'title': metadata['title']
        }
        
    except yt_dlp.utils.DownloadError as e:
        error_msg = f"yt-dlp error: {str(e)}"
        logger.error(f"yt-dlp error for request {url_request_id}: {error_msg}")
        
        # Update URL request status
        url_request.status = TASK_STATES['FAILED_PERMANENTLY']
        url_request.save()
        
        # Handle specific errors
        if 'Video unavailable' in str(e):
            raise self.retry(countdown=120, exc=e)
        elif 'Private video' in str(e):
            raise Exception(f"Private video - cannot extract: {str(e)}")
        else:
            raise self.retry(countdown=60, exc=e)
            
    except URLRequestTable.DoesNotExist:
        logger.error(f"URLRequestTable {url_request_id} not found")
        raise Exception(f"URL request {url_request_id} not found")
        
    except Exception as e:
        logger.error(f"Unexpected error in metadata extraction for {url_request_id}: {str(e)}")
        
        try:
            url_request = URLRequestTable.objects.get(id=url_request_id)
            url_request.status = TASK_STATES['FAILED_PERMANENTLY']
            url_request.save()
        except:
            pass
            
        # Retry on certain errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying metadata extraction for {url_request_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60, exc=e)
        else:
            logger.error(f"Max retries exceeded for metadata extraction {url_request_id}")
            raise e 