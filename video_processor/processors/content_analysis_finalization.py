"""
Phase 2: Content Analysis Finalization Task
Handles timestamp mapping and ratio calculation - runs after embedding is complete.
"""

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction
from django.utils import timezone
import asyncio
from typing import Dict, Any

from api.models import URLRequestTable  
from ..models import ContentAnalysis
from ..config import YOUTUBE_CONFIG
from telemetry import get_logger


# Import the existing functions from the current content_analyzer.py
# We'll reuse these functions for timestamp mapping and ratio calculation
from .content_analyzer import (
    add_timestamps_to_segments,
    calculate_content_ratios
)

logger = get_logger(__name__)


@shared_task(bind=True,
             name='video_processor.content_analysis_finalization',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS'].get('content_analysis_soft_limit', 600),
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS'].get('content_analysis_hard_limit', 900),
             max_retries=2,
             default_retry_delay=60)
def content_analysis_finalization(self, embedding_result, url_request_id):
    """
    Phase 2: Celery task for content analysis finalization.
    
    This task:
    1. Gets ContentAnalysis record with preliminary results (Django ORM - sync)
    2. Runs vector search and ratio calculation (async)
    3. Updates ContentAnalysis with final results (Django ORM - sync)
    
    Runs after embedding is complete to ensure vector search data is available.
    
    Args:
        embedding_result: Result from embedding task
        url_request_id: UUID of the URLRequestTable to process
        
    Returns:
        dict: Final analysis results summary
    """
    content_analysis = None
    video_transcript = None
    
    try:
        logger.info(f"Starting content analysis finalization for request {url_request_id}")
        
        # Check if video was excluded in previous stage
        if embedding_result and embedding_result.get('excluded'):
            logger.info(f"Video was excluded in previous stage: {embedding_result.get('exclusion_reason')}")
            return {
                'video_id': embedding_result.get('video_id'),
                'excluded': True,
                'exclusion_reason': embedding_result.get('exclusion_reason'),
                'skip_reason': 'excluded_in_previous_stage'
            }
        
        # Django ORM operations (sync) - get ContentAnalysis with preliminary results
        url_request = URLRequestTable.objects.select_related(
            'video_metadata',
            'video_metadata__video_transcript',
            'video_metadata__video_transcript__content_analysis'
        ).get(request_id=url_request_id)
        
        if not hasattr(url_request, 'video_metadata') or not url_request.video_metadata:
            raise ValueError("VideoMetadata not found")
        
        video_metadata = url_request.video_metadata
        
        if not hasattr(video_metadata, 'video_transcript') or not video_metadata.video_transcript:
            raise ValueError("VideoTranscript not found")
        
        video_transcript = video_metadata.video_transcript
        video_id = video_transcript.video_id
        
        # Get ContentAnalysis record
        if not hasattr(video_transcript, 'content_analysis') or not video_transcript.content_analysis:
            # Create ContentAnalysis if it doesn't exist (shouldn't happen in normal flow)
            logger.warning(f"ContentAnalysis not found for {video_id}, creating empty record")
            content_analysis = ContentAnalysis.objects.create(
                video_transcript=video_transcript,
                preliminary_analysis_status='failed',  # Mark as failed since preliminary wasn't run
                timestamped_analysis_status='processing'
            )
        else:
            content_analysis = video_transcript.content_analysis
        
        logger.info(f"Processing finalization for video {video_id}: {video_metadata.title}")
        
        # Check if preliminary analysis was completed
        if not content_analysis.is_preliminary_complete:
            logger.warning(f"Preliminary analysis not complete for {video_id}, skipping finalization")
            content_analysis.timestamped_analysis_status = 'failed'
            content_analysis.save(update_fields=['timestamped_analysis_status'])
            
            return {
                'video_id': video_id,
                'finalization_complete': False,
                'error': 'Preliminary analysis not completed',
                'content_rating': 0.0,
                'ad_segments_count': 0,
                'filler_segments_count': 0
            }
        
        # Update status to processing
        with transaction.atomic():
            content_analysis.timestamped_analysis_status = 'processing'
            content_analysis.save(update_fields=['timestamped_analysis_status'])
        
        
        # Run async finalization (isolated from Django ORM)
        final_results = asyncio.run(add_timestamps_and_calculate_ratios_async(
            content_analysis=content_analysis,
            video_id=video_id,
            video_duration=video_metadata.duration or 0
        ))
        # Save final results to ContentAnalysis (sync Django ORM)
        with transaction.atomic():
            content_analysis.ad_segments = final_results['ad_segments']
            content_analysis.filler_segments = final_results['filler_segments']
            content_analysis.content_segments = final_results['content_segments']
            content_analysis.content_rating = final_results['content_rating']
            content_analysis.ad_duration_ratio = final_results['ad_duration_ratio']
            content_analysis.filler_duration_ratio = final_results['filler_duration_ratio']
            content_analysis.timestamped_analysis_status = 'completed'
            content_analysis.final_completed_at = timezone.now()
            content_analysis.save()
        
        logger.info(f"Content analysis finalization complete for {video_id}")
        
        return {
            'video_id': video_id,
            'finalization_complete': True,
            'content_rating': final_results['content_rating'],
            'ad_duration_ratio': final_results['ad_duration_ratio'],
            'filler_duration_ratio': final_results['filler_duration_ratio'],
            'ad_segments_count': len(final_results['ad_segments']),
            'filler_segments_count': len(final_results['filler_segments']),
            'content_segments_count': len(final_results['content_segments']),
            'ads_with_timestamps': final_results['ads_with_timestamps'],
            'fillers_with_timestamps': final_results['fillers_with_timestamps']
        }
        
    except SoftTimeLimitExceeded:
        logger.warning(f"Finalization timeout for request {url_request_id}")
        
        if content_analysis:
            content_analysis.timestamped_analysis_status = 'failed'
            content_analysis.save(update_fields=['timestamped_analysis_status'])
        
        raise
        
    except Exception as e:
        logger.error(f"Finalization failed for request {url_request_id}: {e}")
        
        if content_analysis:
            content_analysis.timestamped_analysis_status = 'failed'
            content_analysis.save(update_fields=['timestamped_analysis_status'])
        
        # Return error result but don't break the chain
        return {
            'video_id': video_transcript.video_id if video_transcript else 'unknown',
            'finalization_complete': False,
            'error': str(e),
            'content_rating': 0.0,
            'ad_segments_count': 0,
            'filler_segments_count': 0,
            'content_segments_count': 0
        }


async def add_timestamps_and_calculate_ratios_async(
    content_analysis: ContentAnalysis, 
    video_id: str, 
    video_duration: int
) -> Dict[str, Any]:
    """
    Phase 2: Add timestamps to raw segments and calculate quality ratios.
    
    This function performs:
    1. Vector search to map text excerpts to timestamps
    2. Quality ratio calculations based on timestamped segments
    
    Args:
        content_analysis: ContentAnalysis instance with preliminary results
        video_id: Video ID for vector search filtering
        video_duration: Total video duration for ratio calculations
        
    Returns:
        dict: Final analysis results with timestamps and ratios
    """
    try:
        logger.info(f"Starting finalization for video {video_id}")
        
        # Step 1: Add timestamps using vector search (reuse existing function)
        logger.info("Adding timestamps to ad segments via vector search")
        timestamped_ads = await add_timestamps_to_segments(
            content_analysis.raw_ad_segments, video_id
        )
        
        logger.info("Adding timestamps to filler segments via vector search") 
        timestamped_filler = await add_timestamps_to_segments(
            content_analysis.raw_filler_segments, video_id
        )
        
        logger.info(f"Vector search complete: {len(timestamped_ads)} ads, {len(timestamped_filler)} fillers with timestamps")
        
        # Calculate timestamp counts
        ads_with_timestamps = sum(1 for seg in timestamped_ads if seg['start'] != 0.0 or seg['end'] != 0.0)
        fillers_with_timestamps = sum(1 for seg in timestamped_filler if seg['start'] != 0.0 or seg['end'] != 0.0)
        
        # Step 2: Calculate quality ratios (reuse existing function)
        timestamped_results = {
            'ad_segments': timestamped_ads,
            'filler_segments': timestamped_filler
        }
        
        ratios = calculate_content_ratios(timestamped_results, video_duration)
        logger.info(f"Calculated ratios: content_rating={ratios['content_rating']:.4f}")
        
        # Calculate content segments (non-ad, non-filler portions)
        content_segments = []
        if video_duration > 0:
            # Simple approach: identify gaps between ad/filler segments as content
            all_segments = []
            
            # Add all timestamped segments
            for seg in timestamped_ads:
                all_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'type': 'ad'
                })
            
            for seg in timestamped_filler:
                all_segments.append({
                    'start': seg['start'], 
                    'end': seg['end'],
                    'type': 'filler'
                })
            
            # Sort by start time and find gaps
            all_segments.sort(key=lambda x: x['start'])
            
            current_time = 0.0
            for segment in all_segments:
                if segment['start'] > current_time + 1.0:  # 1 second minimum gap
                    content_segments.append({
                        'start': current_time,
                        'end': segment['start'],
                        'desc': 'Content segment'
                    })
                current_time = max(current_time, segment['end'])
            
            # Add final content segment if needed
            if current_time < video_duration - 1.0:
                content_segments.append({
                    'start': current_time,
                    'end': video_duration,
                    'desc': 'Content segment'
                })
        
        return {
            'ad_segments': timestamped_ads,
            'filler_segments': timestamped_filler,
            'content_segments': content_segments,
            'content_rating': ratios['content_rating'],
            'ad_duration_ratio': ratios['ad_duration_ratio'],
            'filler_duration_ratio': ratios['filler_duration_ratio'],
            'ads_with_timestamps': ads_with_timestamps,
            'fillers_with_timestamps': fillers_with_timestamps
        }
        
    except Exception as e:
        logger.error(f"Finalization failed for video {video_id}: {e}")
        raise