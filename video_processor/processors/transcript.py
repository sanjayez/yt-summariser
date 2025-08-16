from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.db import transaction
from django.utils import timezone
from celery.exceptions import SoftTimeLimitExceeded
import logging
import time

from api.models import URLRequestTable
from ..models import VideoMetadata, VideoTranscript, TranscriptSegment
from ..config import YOUTUBE_CONFIG, TRANSCRIPT_CONFIG
from ..validators import validate_transcript_data
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task,
    update_task_progress
)
from ..utils.language_detection import (
    detect_transcript_language, get_api_language
)
from ..services.decodo_service import extract_youtube_transcript
from ..services.youtube_transcript_service import extract_youtube_transcript_fallback

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

def extract_transcript_with_fallback(video_id: str, language: str) -> tuple[dict, str, str]:
    """
    Extract transcript using fallback strategy:
    1. Try Decodo API
    2. If fails, try YouTube Transcript API
    3. If both fail, raise exception
    
    Returns:
        Tuple of (transcript_data, used_language, source)
    """
    
    logger.info(f"Starting transcript extraction with fallback for {video_id}")
    
    # Method 1: Try Decodo API (existing)
    if TRANSCRIPT_CONFIG['DECODO']['enabled']:
        try:
            logger.info(f"Attempting Decodo extraction for {video_id}")
            transcript_data, used_language = extract_transcript_with_decodo(video_id, language)
            
            if transcript_data:
                logger.info(f"✅ Decodo extraction successful for {video_id}")
                # Convert Decodo format to our standard format
                result = {
                    'success': True,
                    'transcript_text': ' '.join([segment.get('text', '') for segment in transcript_data]),
                    'segments': transcript_data,
                    'language': used_language,
                    'source': 'decodo',
                    'segment_count': len(transcript_data),
                }
                return result, used_language, 'decodo'
                
        except Exception as e:
            logger.warning(f"Decodo extraction failed for {video_id}: {e}")
    
    # Method 2: Try YouTube Transcript API (fallback)
    if TRANSCRIPT_CONFIG['YOUTUBE_API']['enabled'] and TRANSCRIPT_CONFIG['FALLBACK_STRATEGY']['enable_fallback']:
        try:
            logger.info(f"Attempting YouTube API fallback extraction for {video_id}")
            
            preferred_languages = TRANSCRIPT_CONFIG['YOUTUBE_API']['preferred_languages']
            # Ensure the requested language is first in the list
            if language and language not in preferred_languages:
                preferred_languages = [language] + preferred_languages
            
            success, result = extract_youtube_transcript_fallback(video_id, preferred_languages)
            
            if success and result.get('segments'):
                logger.info(f"✅ YouTube API fallback successful for {video_id}")
                return result, result.get('language', language), 'youtube_api'
                
        except Exception as e:
            logger.warning(f"YouTube API fallback failed for {video_id}: {e}")
    
    # Both methods failed
    logger.error(f"All transcript extraction methods failed for {video_id}")
    raise Exception(f"No transcript available for video {video_id} through any extraction method")

@shared_task(bind=True, 
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['transcript_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['transcript_hard_limit'],
             max_retries=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['max_retries'],
             default_retry_delay=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['countdown'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['jitter'])
@idempotent_task
def extract_video_transcript(self, metadata_result, url_request_id):
    """
    Extract YouTube video transcript using Decodo API.
    
    Args:
        metadata_result (dict): Result from previous metadata extraction task
        url_request_id (str): UUID of the URLRequestTable to process
        
    Returns:
        dict: Transcript extraction results with segments count and language info
        
    Raises:
        Exception: If transcript extraction fails after retries
    """
    progress_recorder = ProgressRecorder(self)
    url_request = None
    
    # Check if video was excluded in previous stage
    if metadata_result and metadata_result.get('excluded'):
        logger.info(f"Video was excluded in metadata stage: {metadata_result.get('exclusion_reason')}")
        # Return immediately without processing transcript
        return {
            'video_id': metadata_result.get('video_id'),
            'excluded': True,
            'exclusion_reason': metadata_result.get('exclusion_reason'),
            'skip_reason': 'excluded_in_metadata_stage'
        }
    video_metadata = None
    video_id = None
    transcript_obj = None
    
    try:
        progress_recorder.set_progress(0, 100, description="Starting transcript extraction")
        
        # Get URL request and video metadata
        url_request = URLRequestTable.objects.get(request_id=url_request_id)
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
        
        # Extract transcript (removed timeout wrapper since it's not functional)
        # Get the video's detected language (using normalized metadata - no fallback needed)
        detected_language = video_metadata.language
        
        # Convert to API-compatible language (fallback-en -> en)
        api_language = get_api_language(detected_language)
        
        # Extract base language code (e.g., 'en' from 'en-US')
        if '-' in api_language:
            base_language = api_language.split('-')[0]
        else:
            base_language = api_language
            
        logger.info(f"Video language: {detected_language}, API language: {api_language}, using base language: {base_language}")
        
        # Use fallback extraction strategy
        transcript_result, used_language, extraction_source = extract_transcript_with_fallback(video_id, base_language)
        
        logger.info(f"Transcript extracted using {extraction_source} for {video_id}")
        logger.info(f"Extracted transcript with {len(transcript_result.get('segments', []))} segments using language: {used_language}")
        
        progress_recorder.set_progress(60, 100, description="Validating transcript data")
        
        # Get transcript data and text from the result
        transcript_data = transcript_result.get('segments', [])
        transcript_text = transcript_result.get('transcript_text', '')
        
        # Validate transcript data if we have segments
        if transcript_data:
            validated_transcript = validate_transcript_data(transcript_data)
            # Only override transcript_text if validation produces different result
            if validated_transcript:
                validated_text = ' '.join([item['text'] for item in validated_transcript])
                if validated_text:
                    transcript_text = validated_text
                    transcript_data = validated_transcript
        
        progress_recorder.set_progress(70, 100, description="Saving transcript to database")
        
        # Save transcript and create segments
        with transaction.atomic():
            transcript_obj.transcript_text = transcript_text
            transcript_obj.status = 'success'
            transcript_obj.transcript_source = extraction_source  # Track source
            transcript_obj.save()
            
            # Language detection for fallback languages only
            if detected_language and detected_language.startswith('fallback-'):
                logger.info(f"Performing language detection for fallback language: {detected_language}")
                is_english, confidence = detect_transcript_language(transcript_text)
                if is_english:
                    updated_language = 'en'
                    video_metadata.language = updated_language
                    video_metadata.save()
                    logger.info(f"Language updated: {detected_language} → {updated_language} (confidence: {confidence:.3f})")
                else:
                    # Non-English content detected - log for classification stage
                    logger.info(f"Non-English content detected (confidence: {confidence:.3f} < threshold)")
                    logger.info(f"Video {video_id} language classification will be handled at final stage")
                    # Note: Language-based exclusion moved to final classification stage
            
            # Clear existing segments (in case of retry)
            TranscriptSegment.objects.filter(transcript=transcript_obj).delete()
            
            # Create transcript segments
            segments_created = []
            for i, segment_data in enumerate(transcript_data):
                segment_id = f"{video_id}_{i+1:03d}"  # e.g., "dQw4w9WgXcQ_001"
                
                # Calculate duration from segment data or next segment
                if 'duration' in segment_data:
                    duration = segment_data['duration']
                elif i+1 < len(transcript_data):
                    duration = transcript_data[i+1]['start'] - segment_data['start']
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
        
        logger.info(f"Successfully extracted transcript with {len(transcript_data)} segments using {extraction_source}")
        logger.info(f"Created {len(segments_created)} transcript segments")
        
        return {
            'transcript_segments': len(transcript_data),
            'segments_created': len(segments_created),
            'language_used': used_language,
            'video_title': metadata_result.get('title', 'Unknown') if isinstance(metadata_result, dict) else video_metadata.title or 'Unknown'
        }
        
    except SoftTimeLimitExceeded:
        # Task is approaching timeout - save status and exit gracefully
        logger.warning(f"Transcript extraction soft timeout reached for video {video_id or 'unknown'}")
        
        try:
            # Update URLRequest with failure reason
            url_request = URLRequestTable.objects.get(request_id=url_request_id)
            url_request.failure_reason = 'no_transcript'
            url_request.save()
            
            # Mark transcript as failed due to timeout
            if video_metadata and video_id:
                VideoTranscript.objects.update_or_create(
                    video_id=video_id,
                    defaults={
                        'video_metadata': video_metadata,
                        'transcript_text': '',
                        'status': 'failed',
                        'transcript_source': 'timeout'
                    }
                )
                logger.error(f"Marked transcript extraction as failed due to timeout: {video_id}")
                
        except Exception as cleanup_error:
            logger.error(f"Failed to update transcript status during timeout cleanup: {cleanup_error}")
        
        # Re-raise with specific timeout message
        raise Exception(f"Transcript extraction timeout for video {video_id or 'unknown'}")
        
    except Exception as e:
        video_id_str = video_id if video_id else 'unknown'
        logger.error(f"Transcript extraction failed for video {video_id_str}: {e}")
        
        # Mark transcript as failed
        with transaction.atomic():
            try:
                # Update URLRequest with failure reason
                url_request = URLRequestTable.objects.get(request_id=url_request_id)
                url_request.failure_reason = 'no_transcript'
                url_request.save()
                
                if not video_metadata:
                    video_metadata = VideoMetadata.objects.get(url_request__id=url_request_id)
                    video_id_str = video_metadata.video_id
                    
                VideoTranscript.objects.update_or_create(
                    video_id=video_id_str,
                    defaults={
                        'video_metadata': video_metadata,
                        'transcript_text': '',
                        'status': 'failed',
                        'transcript_source': 'none'
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