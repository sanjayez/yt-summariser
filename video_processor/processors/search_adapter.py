"""
Video Processing Adapter for Search-to-Process Integration
Adapts the existing video processing pipeline to work with search requests
"""

import logging
from typing import Dict, Any, Optional
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from api.models import URLRequestTable
from topic.models import SearchRequest as SearchRequestModel
from video_processor.models import VideoMetadata, VideoTranscript, update_url_request_status
from video_processor.config import YOUTUBE_CONFIG
from video_processor.processors.workflow import process_youtube_video
from video_processor.validators import validate_youtube_url
from video_processor.utils import handle_dead_letter_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, 
             name='video_processor.process_search_video',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['workflow_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['workflow_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata']['jitter'])
def process_search_video(self, url_request_id: str):
    """
    Video processing adapter for search-generated videos.
    
    This task adapts the existing video processing pipeline to work with
    videos that were generated from search requests, ensuring proper
    linking between SearchRequest → URLRequestTable → VideoMetadata.
    
    Args:
        url_request_id: UUID of the URLRequestTable entry
        
    Returns:
        dict: Processing result with status and details
    """
    logger.info(f"Starting search video processing for URLRequest ID: {url_request_id}")
    
    url_request = None
    
    try:
        # Get URLRequestTable entry with related data
        try:
            url_request = URLRequestTable.objects.select_related(
                'search_request', 
                'search_request__search_session'
            ).get(request_id=url_request_id)  # Use request_id (UUID)
            
            logger.info(f"Processing video {url_request.url} for search request {url_request.search_request.search_id}")
            
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
                'error': 'Invalid URL',
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
        
        # Delegate to main video processing workflow
        try:
            result = process_youtube_video.delay(str(url_request_id))
            logger.info(f"Delegated processing to workflow task {result.id}")
            
            return {
                'status': 'processing',
                'url_request_id': url_request_id,
                'workflow_task_id': result.id,
                'message': 'Video processing started successfully'
            }
            
        except Exception as e:
            error_msg = f"Failed to start video processing workflow: {str(e)}"
            logger.error(error_msg)
            _update_url_request_status(url_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Workflow start failed',
                'details': str(e),
                'url_request_id': url_request_id
            }
            
    except SoftTimeLimitExceeded:
        # Search video processing is approaching timeout
        logger.warning(f"Search video processing soft timeout reached for URLRequest {url_request_id}")
        
        try:
            # Mark as failed due to timeout
            if url_request:
                _update_url_request_status(url_request, 'failed', 'Search video processing timeout')
                logger.error(f"Marked search video processing as failed due to timeout: {url_request_id}")
                
        except Exception as cleanup_error:
            logger.error(f"Failed to update search video status during timeout cleanup: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Search video processing timeout for URLRequest {url_request_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error in search video processing: {e}")
        
        try:
            if url_request:
                _update_url_request_status(url_request, 'failed', f'Processing error: {str(e)}')
        except:
            pass
            
        return {
            'status': 'failed',
            'error': 'Unexpected processing error',
            'details': str(e),
            'url_request_id': url_request_id
        }


@shared_task(bind=True, 
             name='video_processor.update_search_video_status',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'])
def update_search_video_status(self, url_request_id: str, status: str, message: str = None):
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
    
    url_request = None
    
    try:
        # Get URLRequestTable entry
        try:
            url_request = URLRequestTable.objects.select_related(
                'search_request'
            ).get(request_id=url_request_id)
            
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
        
    except SoftTimeLimitExceeded:
        # Status update is approaching timeout - critical issue
        logger.error(f"Status update soft timeout reached for URLRequest {url_request_id} - database issue")
        
        try:
            # Emergency status update
            if url_request:
                url_request.status = 'failed'
                url_request.save()
                logger.error(f"Emergency status update for URLRequest {url_request_id}")
                
        except Exception as cleanup_error:
            logger.critical(f"Failed to emergency update status during timeout: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Status update critical timeout for URLRequest {url_request_id}")
        
    except Exception as e:
        logger.error(f"Error updating search video status: {e}")
        return {
            'status': 'failed',
            'error': 'Failed to update status',
            'details': str(e),
            'url_request_id': url_request_id
        }


@shared_task(bind=True, 
             name='video_processor.get_search_video_results',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'])
def get_search_video_results(self, search_id: str):
    """
    Get processing results for all videos in a search request.
    
    Args:
        search_id: UUID of the SearchRequest
        
    Returns:
        dict: Aggregated results for all videos in the search
    """
    logger.info(f"Getting search video results for search request {search_id}")
    
    search_request = None
    
    try:
        # Get search request
        try:
            search_request = SearchRequestModel.objects.get(search_id=search_id)
        except SearchRequestModel.DoesNotExist:
            error_msg = f"Search request {search_id} not found"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'Search request not found',
                'search_id': search_id
            }
        
        # Get all related URLRequestTable entries with video data
        url_requests = URLRequestTable.objects.filter(
            search_request=search_request
        ).select_related('video_metadata', 'video_metadata__video_transcript')
        
        if not url_requests.exists():
            return {
                'status': 'success',
                'search_id': search_id,
                'total_videos': 0,
                'results': [],
                'message': 'No videos found for this search'
            }
        
        # Aggregate results
        results = []
        for url_request in url_requests:
            video_result = {
                'url_request_id': url_request.request_id,
                'url': url_request.url,
                'status': url_request.status,
                'video_data': None
            }
            
            # Add video metadata if available
            if hasattr(url_request, 'video_metadata'):
                metadata = url_request.video_metadata
                video_result['video_data'] = {
                    'video_id': metadata.video_id,
                    'title': metadata.title,
                    'duration': metadata.duration,
                    'channel_name': metadata.channel_name,
                    'view_count': metadata.view_count,
                    'metadata_status': metadata.status
                }
                
                # Add transcript data if available
                if hasattr(metadata, 'video_transcript'):
                    transcript = metadata.video_transcript
                    video_result['video_data']['transcript_status'] = transcript.status
                    video_result['video_data']['has_summary'] = bool(transcript.summary)
                    video_result['video_data']['transcript_length'] = len(transcript.transcript_text) if transcript.transcript_text else 0
            
            results.append(video_result)
        
        # Calculate summary statistics
        total_videos = len(results)
        successful_videos = sum(1 for r in results if r['status'] == 'success')
        failed_videos = sum(1 for r in results if r['status'] == 'failed')
        processing_videos = sum(1 for r in results if r['status'] == 'processing')
        
        return {
            'status': 'success',
            'search_id': search_id,
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'processing_videos': processing_videos,
            'results': results
        }
        
    except SoftTimeLimitExceeded:
        # Getting results is approaching timeout
        logger.warning(f"Get search video results soft timeout reached for search {search_id}")
        
        # This is a read operation, so just return timeout error
        raise Exception(f"Get search video results timeout for search {search_id}")
        
    except Exception as e:
        logger.error(f"Error getting search video results: {e}")
        return {
            'status': 'failed',
            'error': 'Failed to get results',
            'details': str(e),
            'search_id': search_id
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