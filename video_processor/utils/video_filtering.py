"""
Video Filtering Utilities

This module provides utilities for pre-filtering videos using the VideoExclusionTable
to prevent reprocessing of known problematic videos and improve system efficiency.
"""

import re
import logging
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


def extract_video_id_from_url(video_url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various YouTube URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        str: Video ID if found, None otherwise
    """
    try:
        # Handle youtu.be short URLs
        if 'youtu.be/' in video_url:
            return video_url.split('youtu.be/')[-1].split('?')[0].split('&')[0]
        
        # Handle standard YouTube URLs
        parsed_url = urlparse(video_url)
        
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
            # Extract from query parameter
            if 'watch' in parsed_url.path:
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get('v', [None])[0]
                if video_id:
                    return video_id
            
            # Extract from embed URLs
            elif 'embed' in parsed_url.path:
                return parsed_url.path.split('embed/')[-1].split('?')[0]
        
        # Fallback: regex extraction for video ID pattern
        video_id_pattern = r'[a-zA-Z0-9_-]{11}'
        matches = re.findall(video_id_pattern, video_url)
        if matches:
            return matches[0]
            
    except Exception as e:
        logger.warning(f"Failed to extract video ID from URL {video_url}: {e}")
    
    return None


def is_video_excluded(video_url: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a video is in the exclusion table.
    
    Args:
        video_url: YouTube video URL to check
        
    Returns:
        Tuple[bool, Optional[str]]: (is_excluded, exclusion_reason)
    """
    from video_processor.models import VideoExclusionTable
    
    video_id = extract_video_id_from_url(video_url)
    if not video_id:
        logger.warning(f"Could not extract video ID from URL: {video_url}")
        return False, None
    
    try:
        exclusion_entry = VideoExclusionTable.objects.get(video_id=video_id)
        return True, exclusion_entry.exclusion_reason
    except VideoExclusionTable.DoesNotExist:
        return False, None


def filter_excluded_videos(video_urls: List[str]) -> Tuple[List[str], List[dict]]:
    """
    Filter out excluded videos from a list of video URLs.
    
    Args:
        video_urls: List of YouTube video URLs
        
    Returns:
        Tuple[List[str], List[dict]]: (processable_urls, excluded_info)
            - processable_urls: URLs not in exclusion table
            - excluded_info: List of dicts with excluded video info
    """
    from video_processor.models import VideoExclusionTable
    
    processable_urls = []
    excluded_info = []
    
    # Get all video IDs for batch lookup
    video_id_to_url = {}
    for url in video_urls:
        video_id = extract_video_id_from_url(url)
        if video_id:
            video_id_to_url[video_id] = url
        else:
            logger.warning(f"Could not extract video ID from URL, skipping: {url}")
    
    # Batch lookup of excluded videos for performance
    excluded_entries = VideoExclusionTable.objects.filter(
        video_id__in=video_id_to_url.keys()
    ).values('video_id', 'exclusion_reason')
    
    excluded_video_ids = {entry['video_id']: entry['exclusion_reason'] for entry in excluded_entries}
    
    # Separate processable and excluded URLs
    for video_id, url in video_id_to_url.items():
        if video_id in excluded_video_ids:
            excluded_info.append({
                'url': url,
                'video_id': video_id,
                'exclusion_reason': excluded_video_ids[video_id]
            })
            logger.info(f"Pre-filtered excluded video {video_id}: {excluded_video_ids[video_id]}")
        else:
            processable_urls.append(url)
    
    logger.info(f"Pre-filtering results: {len(processable_urls)} processable, {len(excluded_info)} excluded")
    return processable_urls, excluded_info


def add_video_to_exclusion_table(
    video_url: str, 
    exclusion_reason: str
) -> bool:
    """
    Add a video to the exclusion table.
    
    Args:
        video_url: YouTube video URL
        exclusion_reason: Reason for exclusion (must be valid choice)
        
    Returns:
        bool: True if added successfully, False otherwise
    """
    from video_processor.models import VideoExclusionTable
    
    video_id = extract_video_id_from_url(video_url)
    if not video_id:
        logger.error(f"Cannot add to exclusion table - invalid video URL: {video_url}")
        return False
    
    # Validate exclusion reason
    valid_reasons = [choice[0] for choice in VideoExclusionTable.EXCLUSION_REASONS]
    if exclusion_reason not in valid_reasons:
        logger.error(f"Invalid exclusion reason: {exclusion_reason}. Valid options: {valid_reasons}")
        return False
    
    try:
        exclusion_entry, created = VideoExclusionTable.objects.get_or_create(
            video_id=video_id,
            defaults={
                'video_url': video_url,
                'exclusion_reason': exclusion_reason,
            }
        )
        
        if created:
            logger.info(f"Added video {video_id} to exclusion table: {exclusion_reason}")
            return True
        else:
            logger.debug(f"Video {video_id} already in exclusion table")
            return False
            
    except Exception as e:
        logger.error(f"Failed to add video {video_id} to exclusion table: {e}")
        return False


def get_exclusion_statistics() -> dict:
    """
    Get statistics about excluded videos for analytics.
    
    Returns:
        dict: Statistics about exclusion patterns
    """
    from video_processor.models import VideoExclusionTable
    from django.db.models import Count
    
    try:
        # Get exclusion reason distribution
        reason_stats = VideoExclusionTable.objects.values('exclusion_reason').annotate(
            count=Count('exclusion_reason')
        ).order_by('-count')
        
        total_excluded = VideoExclusionTable.objects.count()
        
        return {
            'total_excluded_videos': total_excluded,
            'exclusion_reasons': list(reason_stats),
            'top_exclusion_reason': reason_stats[0]['exclusion_reason'] if reason_stats else None
        }
    except Exception as e:
        logger.error(f"Failed to get exclusion statistics: {e}")
        return {'error': str(e)} 