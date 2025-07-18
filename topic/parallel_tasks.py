"""
Parallel Processing Orchestrator for Search-to-Process Integration
Handles parallel video processing for search results
"""

import logging
from typing import List, Dict, Any
from celery import shared_task, group, chord
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from api.models import URLRequestTable
from video_processor.processors.workflow import process_youtube_video
from video_processor.processors.search_adapter import process_search_video
from topic.models import SearchRequest as SearchRequestModel
from topic.utils.session_utils import update_session_status

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='topic.process_search_results')
def process_search_results(self, search_id: str):
    """
    Parallel processing orchestrator for search results.
    
    Takes a search_id and processes all found videos in parallel.
    
    Args:
        search_id: UUID of the SearchRequest to process videos for
        
    Returns:
        dict: Processing result with status and details
    """
    logger.info(f"Starting parallel video processing for search request: {search_id}")
    
    try:
        # Get search request from database
        try:
            search_request = SearchRequestModel.objects.select_related('search_session').get(
                search_id=search_id
            )
            session = search_request.search_session
            
            # Validate search request has completed successfully
            if search_request.status != 'success':
                error_msg = f"Search request {search_id} is not in success status (current: {search_request.status})"
                logger.error(error_msg)
                return {
                    'status': 'failed',
                    'error': 'Search request not completed successfully',
                    'search_id': search_id
                }
            
            video_urls = search_request.video_urls
            if not video_urls:
                error_msg = f"No video URLs found for search request {search_id}"
                logger.warning(error_msg)
                return {
                    'status': 'success',
                    'search_id': search_id,
                    'message': 'No videos to process',
                    'processed_videos': 0
                }
            
            logger.info(f"Processing {len(video_urls)} videos for search request {search_id}")
            
        except SearchRequestModel.DoesNotExist:
            error_msg = f"Search request {search_id} not found"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'Search request not found',
                'search_id': search_id
            }
        
        # Create URLRequestTable entries for each video
        try:
            url_request_ids = _create_url_request_entries(search_request, video_urls)
            logger.info(f"Created {len(url_request_ids)} URLRequestTable entries for search request {search_id}")
            
        except Exception as e:
            error_msg = f"Failed to create URLRequestTable entries: {str(e)}"
            logger.error(error_msg)
            _update_search_request_processing_status(search_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Failed to create URL request entries',
                'details': str(e),
                'search_id': search_id
            }
        
        # Update search request status to indicate video processing started
        try:
            _update_search_request_processing_status(search_request, 'processing_videos', 
                                                   f"Started processing {len(url_request_ids)} videos")
            logger.info(f"Updated search request {search_id} to processing_videos status")
            
        except Exception as e:
            logger.error(f"Failed to update search request status: {e}")
            # Continue with processing even if status update fails
        
        # Launch parallel video processing using Celery chord
        try:
            # Create group of video processing tasks
            video_processing_group = group(
                process_youtube_video.s(url_request_id) for url_request_id in url_request_ids
            )
            
            # Use chord to wait for all tasks to complete, then run completion callback
            processing_chord = chord(video_processing_group)(
                finalize_search_processing.s(search_id, url_request_ids)
            )
            
            logger.info(f"Launched parallel processing for {len(url_request_ids)} videos in search request {search_id}")
            
            return {
                'status': 'processing',
                'search_id': search_id,
                'url_request_ids': url_request_ids,
                'total_videos': len(url_request_ids),
                'chord_id': processing_chord.id
            }
            
        except Exception as e:
            error_msg = f"Failed to launch parallel processing: {str(e)}"
            logger.error(error_msg)
            _update_search_request_processing_status(search_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Failed to launch parallel processing',
                'details': str(e),
                'search_id': search_id
            }
        
    except Exception as e:
        logger.error(f"Unexpected error in process_search_results: {e}")
        return {
            'status': 'failed',
            'error': 'Unexpected error',
            'details': str(e),
            'search_id': search_id
        }


@shared_task(bind=True, name='topic.finalize_search_processing')
def finalize_search_processing(self, processing_results: List[Any], search_id: str, url_request_ids: List[int]):
    """
    Completion monitoring system for parallel processing.
    
    Called when all video processing tasks complete.
    Updates search request status based on processing results.
    
    Args:
        processing_results: Results from parallel video processing tasks
        search_id: UUID of the SearchRequest
        url_request_ids: List of URLRequestTable IDs that were processed
        
    Returns:
        dict: Final processing result
    """
    logger.info(f"Finalizing processing for search request {search_id} with {len(url_request_ids)} videos")
    
    try:
        # Get search request from database
        try:
            search_request = SearchRequestModel.objects.select_related('search_session').get(
                search_id=search_id
            )
            session = search_request.search_session
            
        except SearchRequestModel.DoesNotExist:
            error_msg = f"Search request {search_id} not found during finalization"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'Search request not found',
                'search_id': search_id
            }
        
        # Analyze processing results
        processing_stats = _analyze_processing_results(url_request_ids)
        
        # Update search request with final status
        final_status = _determine_final_status(processing_stats)
        status_message = _generate_status_message(processing_stats)
        
        try:
            with transaction.atomic():
                _update_search_request_processing_status(search_request, final_status, status_message)
                
                # Update session status based on search request status
                if final_status == 'completed':
                    update_session_status(session, 'success')
                elif final_status == 'failed':
                    update_session_status(session, 'failed')
                # Keep session as processing if search request is partially_completed
                
                logger.info(f"Search request {search_id} finalized with status: {final_status}")
                
        except Exception as e:
            logger.error(f"Failed to update final status for search request {search_id}: {e}")
        
        return {
            'status': 'completed',
            'search_id': search_id,
            'final_status': final_status,
            'processing_stats': processing_stats,
            'status_message': status_message
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in finalize_search_processing: {e}")
        return {
            'status': 'failed',
            'error': 'Unexpected error during finalization',
            'details': str(e),
            'search_id': search_id
        }


def _create_url_request_entries(search_request: SearchRequestModel, video_urls: List[str]) -> List[int]:
    """
    Create URLRequestTable entries for each video URL.
    
    Args:
        search_request: SearchRequest model instance
        video_urls: List of YouTube video URLs
        
    Returns:
        List[int]: List of URLRequestTable IDs (not UUIDs)
    """
    url_request_entries = []
    
    for url in video_urls:
        url_request_entry = URLRequestTable(
            search_request=search_request,
            url=url,
            ip_address=search_request.search_session.user_ip,
            status='processing'
        )
        url_request_entries.append(url_request_entry)
    
    # Use bulk_create for efficient database operations
    with transaction.atomic():
        created_entries = URLRequestTable.objects.bulk_create(url_request_entries)
        
        # Bulk_create doesn't return IDs in all databases, so we need to fetch them
        # Get the IDs of the created entries by filtering on search_request and creation time
        created_ids = list(
            URLRequestTable.objects.filter(
                search_request=search_request,
                url__in=video_urls
            ).values_list('id', flat=True)  # Use 'id' not 'request_id' for URLRequestTable primary key
        )
    
    return created_ids


def _analyze_processing_results(url_request_ids: List[int]) -> Dict[str, Any]:
    """
    Analyze the results of parallel video processing.
    
    Args:
        url_request_ids: List of URLRequestTable IDs
        
    Returns:
        Dict with processing statistics
    """
    try:
        # Get current status of all URLRequestTable entries
        url_requests = URLRequestTable.objects.filter(id__in=url_request_ids)
        
        total_videos = len(url_request_ids)
        successful_videos = url_requests.filter(status='success').count()
        failed_videos = url_requests.filter(status='failed').count()
        processing_videos = url_requests.filter(status='processing').count()
        
        return {
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'processing_videos': processing_videos,
            'success_rate': (successful_videos / total_videos) * 100 if total_videos > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error analyzing processing results: {e}")
        return {
            'total_videos': len(url_request_ids),
            'successful_videos': 0,
            'failed_videos': 0,
            'processing_videos': len(url_request_ids),
            'success_rate': 0,
            'error': str(e)
        }


def _determine_final_status(processing_stats: Dict[str, Any]) -> str:
    """
    Determine the final status based on processing statistics.
    
    Args:
        processing_stats: Dictionary with processing statistics
        
    Returns:
        str: Final status ('completed', 'partially_completed', or 'failed')
    """
    if processing_stats['processing_videos'] > 0:
        return 'processing'  # Still processing
    elif processing_stats['successful_videos'] == processing_stats['total_videos']:
        return 'completed'  # All successful
    elif processing_stats['successful_videos'] > 0:
        return 'partially_completed'  # Some successful
    else:
        return 'failed'  # All failed


def _generate_status_message(processing_stats: Dict[str, Any]) -> str:
    """
    Generate a human-readable status message.
    
    Args:
        processing_stats: Dictionary with processing statistics
        
    Returns:
        str: Status message
    """
    total = processing_stats['total_videos']
    successful = processing_stats['successful_videos']
    failed = processing_stats['failed_videos']
    processing = processing_stats['processing_videos']
    
    if processing > 0:
        return f"Processing {processing} videos ({successful} completed, {failed} failed)"
    elif successful == total:
        return f"Successfully processed all {total} videos"
    elif successful > 0:
        return f"Processed {successful} of {total} videos successfully ({failed} failed)"
    else:
        return f"Failed to process all {total} videos"


def _update_search_request_processing_status(search_request: SearchRequestModel, status: str, message: str):
    """
    Update SearchRequest with processing status and message.
    
    Args:
        search_request: SearchRequest model instance
        status: Status string
        message: Status message
    """
    try:
        # Store processing info in the error_message field (could be renamed to status_message)
        search_request.error_message = message
        
        # Update status if it's a recognized SearchRequest status
        if status in ['processing', 'failed', 'success']:
            search_request.status = status
        
        search_request.save()
        
    except Exception as e:
        logger.error(f"Failed to update search request status: {e}")


@shared_task(bind=True, name='topic.get_search_processing_status')
def get_search_processing_status(self, search_id: str):
    """
    Get the current status of search result processing.
    
    Args:
        search_id: UUID of the SearchRequest
        
    Returns:
        dict: Current processing status and statistics
    """
    try:
        search_request = SearchRequestModel.objects.select_related('search_session').get(
            search_id=search_id
        )
        
        # Get all related URLRequestTable entries
        url_requests = URLRequestTable.objects.filter(search_request=search_request)
        
        if not url_requests.exists():
            return {
                'status': 'no_processing',
                'search_id': search_id,
                'message': 'No video processing initiated'
            }
        
        # Analyze current processing status
        url_request_ids = list(url_requests.values_list('id', flat=True))
        processing_stats = _analyze_processing_results(url_request_ids)
        
        return {
            'status': 'success',
            'search_id': search_id,
            'processing_stats': processing_stats,
            'search_request_status': search_request.status,
            'status_message': search_request.error_message or 'Processing in progress'
        }
        
    except SearchRequestModel.DoesNotExist:
        return {
            'status': 'failed',
            'error': 'Search request not found',
            'search_id': search_id
        }
    except Exception as e:
        logger.error(f"Error getting search processing status: {e}")
        return {
            'status': 'failed',
            'error': 'Unexpected error',
            'details': str(e),
            'search_id': search_id
        }