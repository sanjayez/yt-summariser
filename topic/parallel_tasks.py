"""
Parallel processing tasks for comprehensive YouTube video analysis.
Handles asynchronous video processing operations with real-time progress tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from django.db import transaction
from celery import shared_task, group
from celery.exceptions import Retry, SoftTimeLimitExceeded

from .models import SearchRequest as SearchRequestModel
from .utils.explorer_progress import ExplorerProgressTracker
from .utils.session_utils import update_session_status
from video_processor.models import URLRequestTable
from video_processor.processors.workflow import process_youtube_video
from video_processor.config import YOUTUBE_CONFIG, BUSINESS_LOGIC_CONFIG
from video_processor.utils import handle_dead_letter_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, 
             name='topic.process_search_results',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['parallel_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['parallel_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['parallel']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['parallel']['jitter'])
def process_search_results(self, search_id: str):
    """
    Parallel processing orchestrator for search results with real-time progress tracking.
    
    Takes a search_id and processes all found videos in parallel (EXCAVATING stage).
    
    Args:
        search_id: UUID of the SearchRequest to process videos for
        
    Returns:
        dict: Processing result with status and details
    """
    # Initialize progress tracker for real-time updates
    progress = ExplorerProgressTracker(search_id)
    
    logger.info(f"Starting parallel video processing for search request: {search_id}")
    
    search_request = None
    session = None
    
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
                progress.send_error(f"Search not ready for processing: {search_request.status}")
                return {
                    'status': 'failed',
                    'error': 'Search request not completed successfully',
                    'search_id': search_id
                }
            
            video_urls = search_request.video_urls
            if not video_urls:
                error_msg = f"No video URLs found for search request {search_id}"
                logger.warning(error_msg)
                # Complete the expedition since there's nothing to process
                progress.expedition_complete()
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
            progress.send_error("Search request not found")
            return {
                'status': 'failed',
                'error': 'Search request not found',
                'search_id': search_id
            }
        
        # 🔍 EXCAVATING STAGE: Processing Video Content
        progress.start_stage('EXCAVATING')
        
        # Strategic delay for stage start
        import time
        time.sleep(1)
        
        # Create URLRequestTable entries for each video
        try:
            # Strategic delay for preparation
            time.sleep(1.5)
            
            url_request_ids = _create_url_request_entries(search_request, video_urls)
            logger.info(f"Created {len(url_request_ids)} URLRequestTable entries for search request {search_id}")
            
        except Exception as e:
            error_msg = f"Failed to create URLRequestTable entries: {str(e)}"
            logger.error(error_msg)
            progress.send_error(f"Failed to prepare video processing: {str(e)}")
            _update_search_request_processing_status(search_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Failed to create URL request entries',
                'details': str(e),
                'search_id': search_id
            }
        
        # Parallel video processing with progress tracking
        try:
            # Strategic delay before starting parallel processing
            time.sleep(1)
            
            # Create Celery group for parallel processing
            job_group = group(
                process_youtube_video.s(url_request_id) 
                for url_request_id in url_request_ids
            )
            
            # Execute parallel processing
            group_result = job_group.apply_async()
            task_ids = [task.id for task in group_result.children or []]
            
            logger.info(f"Initiated parallel processing with {len(task_ids)} tasks for search {search_id}")
            
            # Monitor progress of parallel tasks
            processed_count = 0
            total_videos = len(url_request_ids)
            
            # Poll for completion with progress updates
            while not group_result.ready():
                # Check individual task progress
                completed_tasks = sum(1 for task in group_result.children if task.ready())
                if completed_tasks > processed_count:
                    processed_count = completed_tasks
                    stage_progress = 20 + (processed_count * 60 // total_videos)  # 20-80% range
                    

                    
                    logger.info(f"Video processing progress: {processed_count}/{total_videos} completed")
                
                # Short sleep to avoid busy waiting
                import time
                time.sleep(2)
            

            
            # Count results without calling .get() (which is forbidden in Celery tasks)
            # Instead, check the database status of URL requests
            from video_processor.models import URLRequestTable
            
            completed_count = 0
            successful_count = 0
            failed_count = 0
            
            for url_request_id in url_request_ids:
                try:
                    url_request = URLRequestTable.objects.get(request_id=url_request_id)
                    if url_request.status in ['success', 'failed']:
                        completed_count += 1
                        if url_request.status == 'success':
                            successful_count += 1
                        else:
                            failed_count += 1
                except URLRequestTable.DoesNotExist:
                    logger.warning(f"URL request {url_request_id} not found when checking status")
                    failed_count += 1
            
            logger.info(f"Parallel processing completed: {successful_count} successful, {failed_count} failed, {completed_count} total")
            
        except Exception as e:
            error_msg = f"Parallel video processing failed: {str(e)}"
            logger.error(error_msg)
            progress.send_error(f"Video processing failed: {str(e)}")
            _update_search_request_processing_status(search_request, 'failed', error_msg)
            return {
                'status': 'failed',
                'error': 'Parallel video processing failed',
                'details': str(e),
                'search_id': search_id
            }
        
        # Strategic delay for stage transition
        time.sleep(1)
        
        # 🔬 ANALYZING STAGE: Embedding and Deep Analysis
        progress.start_stage('ANALYZING')
        
        # Strategic delay for embedding and analysis processing
        time.sleep(2)
        
        # 💎 TREASURE_READY STAGE: Final Completion
        progress.start_stage('TREASURE_READY')
        
        # Strategic delay for final processing
        time.sleep(1.5)
        
        # Update final status
        try:
            _update_search_request_processing_status(search_request, 'completed', 
                                                   f"Processed {successful_count} videos successfully")
            update_session_status(session, 'success')
            
            # Strategic delay before final completion
            time.sleep(0.5)
            
            progress.expedition_complete()
            
            logger.info(f"Search results processing completed for {search_id}")
            
            return {
                'status': 'success',
                'search_id': search_id,
                'processed_videos': successful_count,
                'failed_videos': failed_count,
                'total_videos': total_videos,
                'task_ids': task_ids
            }
            
        except Exception as e:
            error_msg = f"Failed to update final status: {str(e)}"
            logger.error(error_msg)
            progress.send_error(f"Failed to finalize results: {str(e)}")
            return {
                'status': 'failed',
                'error': 'Failed to update final status',
                'details': str(e),
                'search_id': search_id
            }
        
    except SoftTimeLimitExceeded:
        # Combined processing is approaching timeout
        logger.warning(f"Combined search and video processing soft timeout reached for search {search_id}")
        progress.send_error("Processing timeout - too many videos or complex content")
        
        try:
            # Mark search as failed due to timeout
            if search_request:
                _update_search_request_processing_status(search_request, 'failed', 
                                                       "Processing timed out - query may be too complex")
            if session:
                update_session_status(session, 'failed')
            logger.error(f"Marked combined processing as failed due to timeout: {search_id}")
            
        except Exception as cleanup_error:
            logger.error(f"Failed to update status during combined processing timeout cleanup: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Combined search and video processing timeout for search {search_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error in parallel processing: {e}")
        progress.send_error(f"Unexpected processing error: {str(e)}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('process_search_results', self.request.id, [search_id], {}, e)
        
        try:
            if search_request:
                _update_search_request_processing_status(search_request, 'failed', f"Unexpected error: {str(e)}")
            if session:
                update_session_status(session, 'failed')
        except:
            pass
            
        return {
            'status': 'failed',
            'error': 'Unexpected error in parallel processing',
            'details': str(e),
            'search_id': search_id
        }


@shared_task(bind=True, 
             name='topic.finalize_search_processing',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'])
def finalize_search_processing(self, processing_results: List[Any], search_id: str, url_request_ids: List[str]):
    """
    Completion monitoring system for parallel processing.
    
    Called when all video processing tasks complete.
    Updates search request status based on processing results.
    
    Args:
        processing_results: Results from parallel video processing tasks
        search_id: UUID of the SearchRequest
        url_request_ids: List of URLRequestTable UUIDs that were processed
        
    Returns:
        dict: Final processing result
    """
    logger.info(f"Finalizing processing for search request {search_id} with {len(url_request_ids)} videos")
    
    search_request = None
    session = None
    
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
                
                # Session status should reflect search success, not video processing results
                # If search found videos (SearchRequest.status = 'success'), keep session as 'success'
                # Only update session to 'failed' if SearchRequest.status is 'failed'
                if search_request.status == 'failed':
                    update_session_status(session, 'failed')
                elif search_request.status == 'success':
                    update_session_status(session, 'success')  # Keep success regardless of video processing
                # Keep session as processing if search is still processing
                
                logger.info(f"Search request {search_id} finalized with video processing status: {final_status}")
                logger.info(f"SearchRequest.status remains: {search_request.status} (search success)")
                
        except Exception as e:
            logger.error(f"Failed to update final status for search request {search_id}: {e}")
        
        return {
            'status': 'completed',
            'search_id': search_id,
            'final_status': final_status,
            'processing_stats': processing_stats,
            'status_message': status_message
        }
        
    except SoftTimeLimitExceeded:
        # Finalization is approaching timeout - critical issue since this should be fast
        logger.error(f"Finalization soft timeout reached for search {search_id} - database issue")
        
        try:
            # Emergency status update
            if search_request:
                search_request.status = 'failed'
                search_request.error_message = 'Finalization timed out - database issue'
                search_request.save()
            if session:
                update_session_status(session, 'failed')
            logger.error(f"Emergency finalization status update for search {search_id}")
            
        except Exception as cleanup_error:
            logger.critical(f"Failed to emergency update status during finalization timeout: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Finalization critical timeout for search {search_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error in finalize_search_processing: {e}")
        return {
            'status': 'failed',
            'error': 'Unexpected error during finalization',
            'details': str(e),
            'search_id': search_id
        }


def _create_url_request_entries(search_request: SearchRequestModel, video_urls: List[str]) -> List[str]:
    """
    Create URLRequestTable entries for each video URL.
    Includes pre-filtering to exclude known problematic videos.
    
    Args:
        search_request: SearchRequest model instance
        video_urls: List of YouTube video URLs
        
    Returns:
        List[str]: List of URLRequestTable UUIDs for processable videos
    """
    from video_processor.utils import filter_excluded_videos
    
    # Pre-filter excluded videos
    processable_urls, excluded_info = filter_excluded_videos(video_urls)
    
    # Log pre-filtering results
    if excluded_info:
        logger.info(f"Pre-filtered {len(excluded_info)} excluded videos for search {search_request.search_id}")
        for exc in excluded_info:
            logger.debug(f"Excluded video {exc['video_id']}: {exc['exclusion_reason']}")
    
    if not processable_urls:
        logger.warning(f"No processable videos after pre-filtering for search {search_request.search_id}")
        return []
    
    logger.info(f"Creating URLRequestTable entries for {len(processable_urls)} processable videos "
                f"({len(excluded_info)} pre-filtered)")
    
    url_request_entries = []
    
    for url in processable_urls:
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
                url__in=processable_urls  # Use processable_urls instead of original video_urls
            ).values_list('request_id', flat=True)  # Use 'request_id' UUID for URLRequestTable
        )
    
    return created_ids


def _analyze_processing_results(url_request_ids: List[str]) -> Dict[str, Any]:
    """
    Analyze the results of parallel video processing.
    
    Args:
        url_request_ids: List of URLRequestTable UUIDs
        
    Returns:
        Dict with processing statistics
    """
    try:
        # Get current status of all URLRequestTable entries
        url_requests = URLRequestTable.objects.filter(request_id__in=url_request_ids)
        
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
    Uses configurable success threshold to determine if partial success qualifies as overall success.
    
    Args:
        processing_stats: Dictionary with processing statistics
        
    Returns:
        str: Final status ('success', 'processing', or 'failed') - matches SearchRequest.STATUS_CHOICES
    """
    from video_processor.config import BUSINESS_LOGIC_CONFIG
    
    if processing_stats['processing_videos'] > 0:
        return 'processing'  # Still processing
    elif processing_stats['successful_videos'] == processing_stats['total_videos']:
        return 'success'  # All successful
    elif processing_stats['successful_videos'] > 0:
        # Check if success rate meets the configurable threshold
        success_rate = processing_stats['success_rate'] / 100  # Convert percentage to decimal
        minimum_threshold = BUSINESS_LOGIC_CONFIG['PIPELINE_SUCCESS']['minimum_threshold']
        
        if success_rate >= minimum_threshold:
            return 'success'  # Meets success threshold (default 85%)
        else:
            return 'failed'   # Below success threshold
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
    Update SearchRequest with processing status message.
    
    NOTE: This function updates the status MESSAGE only, not the SearchRequest.status itself.
    SearchRequest.status should only reflect search success (did we find videos?), 
    not video processing results.
    
    Args:
        search_request: SearchRequest model instance
        status: Processing status string (for message generation only)
        message: Status message about video processing
    """
    try:
        # Store processing info in the error_message field (could be renamed to status_message)
        search_request.error_message = message
        
        # DO NOT update SearchRequest.status here - it should only reflect search success
        # SearchRequest.status is set in process_search_query and should stay 'success' if videos were found
        
        search_request.save()
        
    except Exception as e:
        logger.error(f"Failed to update search request processing message: {e}")


@shared_task(bind=True, 
             name='topic.get_search_processing_status',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'])
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
        
    except SoftTimeLimitExceeded:
        # Status check is approaching timeout
        logger.warning(f"Status check soft timeout reached for search {search_id}")
        
        # This is a read operation, so just return timeout error
        raise Exception(f"Status check timeout for search {search_id}")
        
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