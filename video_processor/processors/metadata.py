import yt_dlp
import os
import json as _json
from urllib.parse import urlencode
from urllib.request import urlopen
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
    update_task_progress, add_video_to_exclusion_table, extract_video_id_from_url
)
from ..utils.metadata_normalizer import normalize_youtube_metadata

logger = logging.getLogger(__name__)

def _fetch_channel_thumbnail_via_youtube_api(channel_id: str) -> str:
    """Return channel avatar thumbnail URL using YouTube Data API v3.
    If anything fails (missing key, HTTP error, parsing), return empty string.
    """
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or not channel_id:
            return ''
        qs = urlencode({
            'part': 'snippet',
            'id': channel_id,
            'key': api_key,
        })
        url = f"https://www.googleapis.com/youtube/v3/channels?{qs}"
        with urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
        items = (data or {}).get('items') or []
        if not items:
            return ''
        thumbs = (((items[0] or {}).get('snippet') or {}).get('thumbnails') or {})
        # Prefer higher quality if available
        for key in ['maxres', 'high', 'medium', 'standard', 'default']:
            if isinstance(thumbs.get(key), dict) and thumbs[key].get('url'):
                return thumbs[key]['url']
        return ''
    except Exception:
        return ''

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

def _handle_metadata_business_failure(url_request: URLRequestTable, error_message: str):
    """
    Handle business logic failures for metadata extraction.
    
    Classifies the failure and adds to VideoExclusionTable if it's a business failure
    (not a technical failure that could be retried).
    
    Args:
        url_request: URLRequestTable instance
        error_message: Error message from the failure
    """
    video_url = url_request.url
    exclusion_reason = None
    
    # Classify the failure reason based on error message
    # Only add to exclusion table for PERMANENT business logic failures
    error_lower = error_message.lower()
    
    if any(keyword in error_lower for keyword in ['private video', 'video unavailable', 'restricted']):
        exclusion_reason = 'content_unavailable'
    elif any(keyword in error_lower for keyword in ['age restricted', 'sign in to confirm']):
        exclusion_reason = 'privacy_restricted'
    elif any(keyword in error_lower for keyword in ['video too short', 'duration too short']):
        exclusion_reason = 'duration_too_short'
    elif any(keyword in error_lower for keyword in ['video too long', 'duration too long', 'video exceeds', 'duration exceeds']):
        exclusion_reason = 'duration_too_long'
    else:
        # For unclear/technical failures, DON'T add to exclusion table
        # Let retry mechanism handle temporary issues
        exclusion_reason = None
        logger.warning(f"Metadata extraction failed with unclear error (not adding to exclusion): {error_message}")
    
    # Add to exclusion table
    if exclusion_reason:
        try:
            added = add_video_to_exclusion_table(video_url, exclusion_reason)
            if added:
                video_id = extract_video_id_from_url(video_url)
                logger.info(f"Added video {video_id} to exclusion table: {exclusion_reason}")
        except Exception as e:
            logger.warning(f"Failed to add video to exclusion table: {e}")
    
    # Update URLRequest status to 'failed' and set appropriate failure reason
    url_request.status = 'failed'
    if exclusion_reason:
        # This is a business exclusion, not a metadata extraction failure
        url_request.failure_reason = 'excluded'
    else:
        # This is an actual metadata extraction failure
        url_request.failure_reason = 'no_metadata'
    url_request.save()

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
        url_request_id (str): UUID of the URLRequestTable to process
        
    Returns:
        dict: Video metadata with video_id and title
        
    Raises:
        Exception: If metadata extraction fails after retries
    """
    try:
        
        # Get the URL request
        url_request = URLRequestTable.objects.get(request_id=url_request_id)
        
        logger.info(f"Starting metadata extraction for request {url_request_id}")
        logger.info(f"Extracting metadata for {url_request.url}")
        
        # Redis progress tracking for SearchProgressAggregator
        update_task_progress(self, 'extracting_metadata', 10)
        
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
        
        # Normalize metadata using the centralized normalization layer
        metadata = normalize_youtube_metadata(info)
        
        # Validate metadata (now using normalized data)
        validate_video_info(metadata)
        
        # Extract language information (exclusion will be handled after classification)
        video_language = metadata.get('language', 'en')
        # Save to database with transaction (using normalized data)
        with transaction.atomic():
            # Fetch channel avatar via YouTube Data API (single call, no retries)
            yt_avatar = _fetch_channel_thumbnail_via_youtube_api(metadata.get('channel_id')) if metadata.get('channel_id') else ''

            video_metadata = VideoMetadata.objects.create(
                video_id=metadata['video_id'],
                title=metadata['title'],
                description=metadata['description'],
                duration=metadata['duration'],
                channel_name=metadata['channel_name'],
                view_count=metadata['view_count'],
                upload_date=metadata['upload_date'],
                language=metadata['language'],
                like_count=metadata['like_count'],
                channel_id=metadata['channel_id'],
                tags=metadata['tags'],
                categories=metadata['categories'],
                thumbnail=metadata['thumbnail'],
                channel_follower_count=metadata['channel_follower_count'],
                channel_is_verified=metadata['channel_is_verified'],
                uploader_id=metadata['uploader_id'],
                comment_count=metadata['comment_count'],
                engagement=metadata['engagement'],
                channel_thumbnail=yt_avatar or '',
                url_request=url_request,
                status='success'
            )
            
            # Note: Music content classification moved to final classification stage
            # Video metadata extraction completed successfully
        
        logger.info(f"Metadata extraction completed successfully for video {metadata['video_id']}")
        
        # Redis progress tracking - metadata complete
        update_task_progress(self, 'extracting_metadata', 100)
        
        # Return the expected format for workflow chain
        return {
            'video_id': metadata['video_id'], 
            'title': metadata['title']
        }
        
    except yt_dlp.utils.DownloadError as e:
        error_msg = f"yt-dlp error: {str(e)}"
        logger.error(f"yt-dlp error for request {url_request_id}: {error_msg}")
        
        # Handle business failure and update status
        _handle_metadata_business_failure(url_request, error_msg)
        
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
            url_request = URLRequestTable.objects.get(request_id=url_request_id)
            _handle_metadata_business_failure(url_request, str(e))
        except:
            pass
            
        # Retry on certain errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying metadata extraction for {url_request_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60, exc=e)
        else:
            logger.error(f"Max retries exceeded for metadata extraction {url_request_id}")
            raise e 