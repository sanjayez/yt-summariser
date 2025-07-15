from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.db import transaction
import logging
import time

from api.models import URLRequestTable
from ..models import VideoMetadata, VideoTranscript, TranscriptSegment
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..validators import validate_transcript_data
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)
from ..services.decodo_service import extract_youtube_transcript

logger = logging.getLogger(__name__)

def get_language_variants(primary_language):
    """Get common language variants for a primary language code"""
    variants = {
        'en': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],
        'es': ['es', 'es-ES', 'es-MX', 'es-AR'],
        'fr': ['fr', 'fr-FR', 'fr-CA'],
        'de': ['de', 'de-DE', 'de-AT'],
        'it': ['it', 'it-IT'],
        'pt': ['pt', 'pt-BR', 'pt-PT'],
        'zh': ['zh', 'zh-CN', 'zh-TW'],
        'ja': ['ja', 'ja-JP'],
        'ko': ['ko', 'ko-KR'],
        'ru': ['ru', 'ru-RU'],
        'ar': ['ar', 'ar-SA'],
        'hi': ['hi', 'hi-IN'],
    }
    return variants.get(primary_language, [primary_language])

def extract_transcript_with_decodo(video_id, primary_language='en'):
    """
    Extract transcript using Decodo API with language fallback support
    
    Args:
        video_id: YouTube video ID
        primary_language: Primary language to try first
        
    Returns:
        Tuple of (transcript_data, used_language)
    """
    logger.info(f"Extracting transcript for video {video_id} using Decodo API")
    
    # Try primary language first
    success, result = extract_youtube_transcript(video_id, primary_language)
    
    if success and result.get('segments'):
        logger.info(f"Successfully extracted transcript using language: {primary_language}")
        return result['segments'], primary_language
    
    # Try language variants if primary fails
    language_variants = get_language_variants(primary_language)
    
    for lang_code in language_variants[1:]:  # Skip first one (already tried)
        try:
            logger.info(f"Trying language variant: {lang_code}")
            success, result = extract_youtube_transcript(video_id, lang_code)
            
            if success and result.get('segments'):
                logger.info(f"Successfully extracted transcript using language variant: {lang_code}")
                return result['segments'], lang_code
        except Exception as e:
            logger.debug(f"Failed with language {lang_code}: {e}")
            continue
    
    # Try common fallback languages
    fallback_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
    remaining_languages = [lang for lang in fallback_languages if lang not in language_variants]
    
    for lang_code in remaining_languages:
        try:
            logger.info(f"Trying fallback language: {lang_code}")
            success, result = extract_youtube_transcript(video_id, lang_code)
            
            if success and result.get('segments'):
                logger.info(f"Successfully extracted transcript using fallback language: {lang_code}")
                return result['segments'], lang_code
        except Exception as e:
            logger.debug(f"Failed with fallback language {lang_code}: {e}")
            continue
    
    # If all languages fail, raise exception
    raise Exception(f"No transcript found for video {video_id} in any supported language")

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
@idempotent_task
def extract_video_transcript(self, metadata_result, url_request_id):
    """
    Extract YouTube video transcript using Decodo API.
    
    Args:
        metadata_result (dict): Result from previous metadata extraction task
        url_request_id (int): ID of the URLRequestTable to process
        
    Returns:
        dict: Transcript extraction results with segments count and language info
        
    Raises:
        Exception: If transcript extraction fails after retries
    """
    progress_recorder = ProgressRecorder(self)
    
    try:
        progress_recorder.set_progress(0, 100, description="Starting transcript extraction")
        
        # Get URL request and video metadata
        url_request = URLRequestTable.objects.get(id=url_request_id)
        video_metadata = VideoMetadata.objects.get(url_request=url_request)
        video_id = video_metadata.video_id
        
        logger.info(f"Starting transcript extraction for video {video_id} using Decodo API")
        
        progress_recorder.set_progress(10, 100, description="Initializing transcript extraction")
        
        # Create or get transcript record
        with transaction.atomic():
            transcript_obj, created = VideoTranscript.objects.get_or_create(
                video_id=video_id,
                defaults={
                    'video_metadata': video_metadata,
                    'transcript_text': '',
                    'status': 'processing'
                }
            )
            
            if not created:
                transcript_obj.status = 'processing'
                transcript_obj.save()
        
        progress_recorder.set_progress(20, 100, description="Extracting transcript from Decodo API")
        
        # Extract transcript with timeout protection
        with timeout(YOUTUBE_CONFIG['TASK_TIMEOUTS']['transcript_timeout'], "Transcript extraction"):
            # Get the video's detected language or use default
            detected_language = getattr(video_metadata, 'language', 'en') or 'en'
            
            # Extract base language code (e.g., 'en' from 'en-US')
            if '-' in detected_language:
                base_language = detected_language.split('-')[0]
            else:
                base_language = detected_language
                
            logger.info(f"Video language: {detected_language}, using base language: {base_language}")
            transcript_data, used_language = extract_transcript_with_decodo(video_id, base_language)
            
            logger.info(f"Extracted transcript with {len(transcript_data)} segments using language: {used_language}")
        
        progress_recorder.set_progress(60, 100, description="Validating transcript data")
        
        # Validate transcript data
        validated_transcript = validate_transcript_data(transcript_data)
        transcript_text = ' '.join([item['text'] for item in validated_transcript])
        
        progress_recorder.set_progress(70, 100, description="Saving transcript to database")
        
        # Save transcript and create segments
        with transaction.atomic():
            transcript_obj.transcript_text = transcript_text
            transcript_obj.status = 'success'
            transcript_obj.save()
            
            # Clear existing segments (in case of retry)
            TranscriptSegment.objects.filter(transcript=transcript_obj).delete()
            
            # Create transcript segments
            segments_created = []
            for i, segment_data in enumerate(validated_transcript):
                segment_id = f"{video_id}_{i+1:03d}"  # e.g., "dQw4w9WgXcQ_001"
                
                # Calculate duration from segment data or next segment
                if 'duration' in segment_data:
                    duration = segment_data['duration']
                elif i+1 < len(validated_transcript):
                    duration = validated_transcript[i+1]['start'] - segment_data['start']
                else:
                    duration = 5.0  # Default 5 seconds for last segment
                
                segment_obj = TranscriptSegment.objects.create(
                    transcript=transcript_obj,
                    segment_id=segment_id,
                    sequence_number=i+1,
                    start_time=segment_data['start'],
                    duration=duration,
                    text=segment_data['text'],
                    is_embedded=False,
                )
                
                segments_created.append(segment_id)
        
        progress_recorder.set_progress(100, 100, description="Transcript extraction complete")
        
        logger.info(f"Successfully extracted transcript with {len(validated_transcript)} segments using Decodo API")
        logger.info(f"Created {len(segments_created)} transcript segments")
        
        return {
            'transcript_segments': len(validated_transcript),
            'segments_created': len(segments_created),
            'language_used': used_language,
            'video_title': metadata_result.get('title', 'Unknown') if isinstance(metadata_result, dict) else video_metadata.title or 'Unknown'
        }
        
    except Exception as e:
        video_id_str = video_id if 'video_id' in locals() else 'unknown'
        logger.error(f"Transcript extraction failed for video {video_id_str}: {e}")
        
        # Mark transcript as failed
        with transaction.atomic():
            try:
                video_metadata = VideoMetadata.objects.get(url_request__id=url_request_id)
                VideoTranscript.objects.update_or_create(
                    video_id=video_metadata.video_id,
                    defaults={
                        'video_metadata': video_metadata,
                        'transcript_text': '',
                        'status': 'failed'
                    }
                )
            except Exception as db_error:
                logger.error(f"Failed to update transcript status: {db_error}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('extract_video_transcript', self.request.id, [url_request_id], {}, e)
        
        # Return error result but don't break the chain (graceful degradation)
        return {
            'transcript_segments': 0, 
            'segments_created': 0, 
            'language_used': 'unknown',
            'video_title': 'Unknown', 
            'error': str(e)
        } 