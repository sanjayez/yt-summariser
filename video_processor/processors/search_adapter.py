"""
Video Processing Adapter for Search-to-Process Integration
Adapts the existing video processing pipeline to work with search requests
"""

import logging
from typing import Dict, Any, Optional
from celery import shared_task
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from api.models import URLRequestTable
from topic.models import SearchRequest as SearchRequestModel
from video_processor.models import VideoMetadata, VideoTranscript, update_url_request_status
from video_processor.processors.workflow import process_youtube_video
from video_processor.validators import validate_youtube_url

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='video_processor.process_search_video')
def process_search_video(self, url_request_id: int):
    """
    Video processing adapter for search-generated videos.
    
    This task adapts the existing video processing pipeline to work with
    videos that were generated from search requests, ensuring proper
    linking between SearchRequest → URLRequestTable → VideoMetadata.
    
    Args:
        url_request_id: ID (not UUID) of the URLRequestTable entry
        
    Returns:
        dict: Processing result with status and details
    """
    logger.info(f"Starting search video processing for URLRequest ID: {url_request_id}")
    
    try:
        # Get URLRequestTable entry with related data
        try:
            url_request = URLRequestTable.objects.select_related(
                'search_request', 
                'search_request__search_session'
            ).get(id=url_request_id)  # Use id (int) not request_id (UUID)
            
            logger.info(f"Processing video {url_request.url} for search request {url_request.search_request.request_id}")
            
        except URLRequestTable.DoesNotExist:
            error_msg = f"URLRequestTable entry {url_request_id} not found"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'URL request not found',
                'url_request_id': url_request_id
            }
        
        # Validate YouTube URL
        try:
            validate_youtube_url(url_request.url)
        except Exception as e:
            error_msg = f"Invalid YouTube URL: {str(e)}"
            logger.error(error_msg)
            _update_url_request_status(url_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Invalid YouTube URL',
                'details': str(e),
                'url_request_id': url_request_id
            }
        
        # Check if video is already processed
        try:
            existing_video = VideoMetadata.objects.get(url_request=url_request)
            logger.info(f"Video already exists for URLRequest {url_request_id}, status: {existing_video.status}")
            
            # If already successful, return success
            if existing_video.status == 'success':
                return {
                    'status': 'success',
                    'url_request_id': url_request_id,
                    'video_id': existing_video.video_id,
                    'message': 'Video already processed successfully'
                }
            # If failed, we'll retry the processing
            elif existing_video.status == 'failed':
                logger.info(f"Retrying failed video processing for URLRequest {url_request_id}")
                
        except VideoMetadata.DoesNotExist:
            # New video, continue with processing
            logger.info(f"New video processing for URLRequest {url_request_id}")
        
        # Update URL request status to processing
        _update_url_request_status(url_request, 'processing', 'Started video processing')
        
        # Call the existing video processing workflow
        # This will handle the chain: metadata → transcript → summary → embedding → status
        try:
            result = process_youtube_video.apply_async(args=[url_request_id])
            
            logger.info(f"Video processing pipeline initiated for URLRequest {url_request_id}, task ID: {result.id}")
            
            return {
                'status': 'processing',
                'url_request_id': url_request_id,
                'task_id': result.id,
                'search_request_id': str(url_request.search_request.request_id),
                'video_url': url_request.url
            }
            
        except Exception as e:
            error_msg = f"Failed to start video processing pipeline: {str(e)}"
            logger.error(error_msg)
            _update_url_request_status(url_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Failed to start video processing',
                'details': str(e),
                'url_request_id': url_request_id
            }
        
    except Exception as e:
        logger.error(f"Unexpected error in process_search_video: {e}")
        return {
            'status': 'failed',
            'error': 'Unexpected error',
            'details': str(e),
            'url_request_id': url_request_id
        }


@shared_task(bind=True, name='video_processor.update_search_video_status')
def update_search_video_status(self, url_request_id: int, status: str, message: str = None):
    """
    Update the status of a video processing task for search-generated videos.
    
    This task handles status updates and ensures proper linking between
    the video processing results and the search request.
    
    Args:
        url_request_id: ID of the URLRequestTable entry
        status: New status ('processing', 'success', 'failed')
        message: Optional status message
        
    Returns:
        dict: Update result
    """
    logger.info(f"Updating search video status for URLRequest {url_request_id}: {status}")
    
    try:
        # Get URLRequestTable entry
        try:
            url_request = URLRequestTable.objects.select_related(
                'search_request'
            ).get(id=url_request_id)
            
        except URLRequestTable.DoesNotExist:
            error_msg = f"URLRequestTable entry {url_request_id} not found"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'URL request not found',
                'url_request_id': url_request_id
            }
        
        # Update URL request status
        _update_url_request_status(url_request, status, message)
        
        # If this is a final status update, check if we should update search request status
        if status in ['success', 'failed']:
            _check_and_update_search_request_status(url_request.search_request)
        
        logger.info(f"Successfully updated URLRequest {url_request_id} status to {status}")
        
        return {
            'status': 'success',
            'url_request_id': url_request_id,
            'updated_status': status,
            'message': message
        }
        
    except Exception as e:
        logger.error(f"Error updating search video status: {e}")
        return {
            'status': 'failed',
            'error': 'Failed to update status',
            'details': str(e),
            'url_request_id': url_request_id
        }


@shared_task(bind=True, name='video_processor.get_search_video_results')
def get_search_video_results(self, search_request_id: str):
    """
    Get the results of video processing for a search request.
    
    Args:
        search_request_id: UUID of the SearchRequest
        
    Returns:
        dict: Video processing results and metadata
    """
    logger.info(f"Getting search video results for search request: {search_request_id}")
    
    try:
        # Get search request
        try:
            search_request = SearchRequestModel.objects.get(request_id=search_request_id)
            
        except SearchRequestModel.DoesNotExist:
            return {
                'status': 'failed',
                'error': 'Search request not found',
                'search_request_id': search_request_id
            }
        
        # Get all related URLRequestTable entries with video data
        url_requests = URLRequestTable.objects.filter(
            search_request=search_request
        ).select_related('video_metadata').prefetch_related(
            'video_metadata__video_transcript'
        )
        
        # Compile results
        video_results = []
        for url_request in url_requests:
            video_data = {
                'url_request_id': url_request.id,
                'url': url_request.url,
                'status': url_request.status,
                'video_metadata': None,
                'video_transcript': None
            }
            
            # Add video metadata if available
            if hasattr(url_request, 'video_metadata') and url_request.video_metadata:
                metadata = url_request.video_metadata
                video_data['video_metadata'] = {
                    'video_id': metadata.video_id,
                    'title': metadata.title,
                    'description': metadata.description,
                    'duration': metadata.duration,
                    'channel_name': metadata.channel_name,
                    'view_count': metadata.view_count,
                    'upload_date': metadata.upload_date.isoformat() if metadata.upload_date else None,
                    'thumbnail': metadata.thumbnail,
                    'status': metadata.status,
                    'is_embedded': metadata.is_embedded
                }
                
                # Add transcript data if available
                transcript = metadata.video_transcript
                if transcript:
                    video_data['video_transcript'] = {
                        'video_id': transcript.video_id,
                        'summary': transcript.summary,
                        'key_points': transcript.key_points,
                        'status': transcript.status,
                        'transcript_available': bool(transcript.transcript_text)
                    }
            
            video_results.append(video_data)
        
        # Calculate summary statistics
        total_videos = len(video_results)
        successful_videos = sum(1 for v in video_results if v['status'] == 'success')
        failed_videos = sum(1 for v in video_results if v['status'] == 'failed')
        processing_videos = sum(1 for v in video_results if v['status'] == 'processing')
        
        return {
            'status': 'success',
            'search_request_id': search_request_id,
            'video_results': video_results,
            'summary': {
                'total_videos': total_videos,
                'successful_videos': successful_videos,
                'failed_videos': failed_videos,
                'processing_videos': processing_videos,
                'success_rate': (successful_videos / total_videos) * 100 if total_videos > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting search video results: {e}")
        return {
            'status': 'failed',
            'error': 'Failed to get video results',
            'details': str(e),
            'search_request_id': search_request_id
        }


def _update_url_request_status(url_request: URLRequestTable, status: str, message: str = None):
    """
    Update URLRequestTable status.
    
    Args:
        url_request: URLRequestTable instance
        status: New status
        message: Optional status message
    """
    try:
        url_request.status = status
        url_request.save()
        
        # Also update using the model's helper function to ensure consistency
        update_url_request_status(url_request)
        
    except Exception as e:
        logger.error(f"Failed to update URLRequest status: {e}")


def _check_and_update_search_request_status(search_request: SearchRequestModel):
    """
    Check if all videos in a search request are processed and update search request status.
    
    Args:
        search_request: SearchRequest instance
    """
    try:
        # Get all related URLRequestTable entries
        url_requests = URLRequestTable.objects.filter(search_request=search_request)
        
        if not url_requests.exists():
            return
        
        # Count statuses
        total_videos = url_requests.count()
        successful_videos = url_requests.filter(status='success').count()
        failed_videos = url_requests.filter(status='failed').count()
        processing_videos = url_requests.filter(status='processing').count()
        
        # Update search request status based on video processing results
        if processing_videos > 0:
            # Still processing
            status_message = f"Processing {processing_videos} videos ({successful_videos} completed, {failed_videos} failed)"
        elif successful_videos == total_videos:
            # All successful
            search_request.status = 'success'
            status_message = f"Successfully processed all {total_videos} videos"
        elif successful_videos > 0:
            # Partially successful
            status_message = f"Processed {successful_videos} of {total_videos} videos successfully ({failed_videos} failed)"
        else:
            # All failed
            search_request.status = 'failed'
            status_message = f"Failed to process all {total_videos} videos"
        
        # Update search request with status message
        search_request.error_message = status_message
        search_request.save()
        
        logger.info(f"Updated search request {search_request.request_id} status: {status_message}")
        
    except Exception as e:
        logger.error(f"Failed to update search request status: {e}")