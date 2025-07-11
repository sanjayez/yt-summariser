from celery import shared_task
from youtube_transcript_api import YouTubeTranscriptApi
from django.db import transaction
import logging
import re

from api.models import URLRequestTable
from ..models import VideoMetadata, VideoTranscript, TranscriptSegment
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..validators import validate_transcript_data
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)

def get_language_variants(primary_language):
    """Get common language variants for a primary language code"""
    variants = {
        'en': ['en', 'en-GB', 'en-US', 'en-CA', 'en-AU', 'en-NZ'],
        'es': ['es', 'es-ES', 'es-MX', 'es-AR', 'es-CO'],
        'fr': ['fr', 'fr-FR', 'fr-CA'],
        'de': ['de', 'de-DE', 'de-AT', 'de-CH'],
        'it': ['it', 'it-IT'],
        'pt': ['pt', 'pt-BR', 'pt-PT'],
        'zh': ['zh', 'zh-CN', 'zh-TW', 'zh-HK'],
        'ja': ['ja', 'ja-JP'],
        'ko': ['ko', 'ko-KR'],
        'ru': ['ru', 'ru-RU'],
        'ar': ['ar', 'ar-SA', 'ar-EG'],
        'hi': ['hi', 'hi-IN'],
    }
    return variants.get(primary_language, [primary_language])

def parse_available_languages(error_message):
    """Parse available languages from the YouTube transcript API error message"""
    # Look for lines like: ' - en-GB ("English (United Kingdom)")'
    pattern = r'- ([a-zA-Z\-]+) \("([^"]+)"\)'
    matches = re.findall(pattern, error_message)
    return [match[0] for match in matches]

def extract_transcript_with_fallback(video_id, primary_language='en'):
    """
    Extract transcript with language fallback support.
    Tries primary language variants, then available languages from error message.
    """
    # Try primary language variants first
    language_variants = get_language_variants(primary_language)
    
    for lang_code in language_variants:
        try:
            logger.info(f"Trying transcript extraction with language: {lang_code}")
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
            logger.info(f"Successfully extracted transcript using language: {lang_code}")
            return transcript_data, lang_code
        except Exception as e:
            logger.debug(f"Failed to extract transcript with {lang_code}: {e}")
            continue
    
    # If primary variants fail, try to extract available languages from error
    try:
        # This will fail, but we want to capture the error message
        YouTubeTranscriptApi.get_transcript(video_id, languages=[primary_language])
    except Exception as e:
        error_message = str(e)
        available_languages = parse_available_languages(error_message)
        
        if available_languages:
            logger.info(f"Found available languages from error: {available_languages}")
            
            # Try each available language
            for lang_code in available_languages:
                try:
                    logger.info(f"Trying available language: {lang_code}")
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
                    logger.info(f"Successfully extracted transcript using available language: {lang_code}")
                    return transcript_data, lang_code
                except Exception as lang_error:
                    logger.debug(f"Failed to extract transcript with available language {lang_code}: {lang_error}")
                    continue
        
        # If all attempts fail, re-raise the original error
        raise e
    
    # This shouldn't be reached, but just in case
    raise Exception(f"No transcripts available for video {video_id}")

# Task 2: Extract Video Transcript
@shared_task(bind=True,
             name='video_processor.extract_video_transcript',
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['jitter'],
             retry_kwargs=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript'])
@idempotent_task
def extract_video_transcript(self, metadata_result, url_request_id):
    """
    Atomic task: Extract transcript with timeout protection and validation.
    Creates transcript segments with meaningful IDs for vector embedding.
    """
    try:
        video_id = metadata_result.get('video_id') if isinstance(metadata_result, dict) else None
        
        if not video_id:
            raise ValueError("No video_id received from metadata extraction")
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 10)
        
        # Get VideoMetadata object
        video_metadata = VideoMetadata.objects.get(
            url_request__id=url_request_id
        )
        
        logger.info(f"Extracting transcript for video {video_id}")
        
        # Create transcript object with transaction
        with transaction.atomic():
            transcript_obj, created = VideoTranscript.objects.get_or_create(
                video_metadata=video_metadata,  # Use the foreign key for lookup
                defaults={
                    'video_id': video_id,  # Set the primary key
                    'transcript_text': '',
                    'status': 'processing'
                }
            )
            
            if not created:
                transcript_obj.status = 'processing'
                transcript_obj.save()
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 30)
        
        # Extract transcript with timeout protection and language fallback
        with timeout(YOUTUBE_CONFIG['TASK_TIMEOUTS']['transcript_timeout'], "Transcript extraction"):
            # Get the video's detected language or use default
            detected_language = getattr(video_metadata, 'language', 'en') or 'en'
            transcript_data, used_language = extract_transcript_with_fallback(video_id, detected_language)
            
            logger.info(f"Extracted transcript using language: {used_language}")
        
        # Validate transcript data
        validated_transcript = validate_transcript_data(transcript_data)
        transcript_text = ' '.join([item['text'] for item in validated_transcript])
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 70)
        
        # Save with transaction and create segments
        with transaction.atomic():
            transcript_obj.transcript_text = transcript_text
            transcript_obj.status = 'success'
            transcript_obj.save()
            
            # Create transcript segments with meaningful IDs
            segments_created = []
            for i, segment_data in enumerate(validated_transcript):
                segment_id = f"{video_id}_{i+1:03d}"  # e.g., "dQw4w9WgXcQ_001"
                
                # Calculate duration (next segment's start - current start, or use default)
                if i+1 < len(validated_transcript):
                    duration = validated_transcript[i+1]['start'] - segment_data['start']
                else:
                    duration = segment_data.get('duration', 5)  # Default 5 seconds for last segment
                
                segment_obj, created = TranscriptSegment.objects.get_or_create(
                    transcript=transcript_obj,
                    sequence_number=i+1,
                    defaults={
                        'segment_id': segment_id,
                        'start_time': segment_data['start'],
                        'duration': duration,
                        'text': segment_data['text'],
                        'is_embedded': False,
                    }
                )
                
                if created:
                    segments_created.append(segment_id)
        
        update_task_progress(self, TASK_STATES['EXTRACTING_TRANSCRIPT'], 100)
        
        logger.info(f"Successfully extracted transcript with {len(validated_transcript)} segments")
        logger.info(f"Created {len(segments_created)} transcript segments with IDs: {segments_created[:5]}...")
        
        return {
            'transcript_segments': len(validated_transcript),
            'segments_created': len(segments_created),
            'video_title': metadata_result.get('title', 'Unknown') if isinstance(metadata_result, dict) else 'Unknown'
        }
        
    except Exception as e:
        logger.warning(f"Transcript extraction failed for video {video_id}: {e}")
        
        # Mark transcript as failed but don't stop the chain
        with transaction.atomic():
            try:
                video_metadata = VideoMetadata.objects.get(url_request__id=url_request_id)
                VideoTranscript.objects.update_or_create(
                    video_metadata=video_metadata,  # Use the foreign key for lookup
                    defaults={
                        'video_id': video_metadata.video_id,  # Set the primary key
                        'transcript_text': '',
                        'status': 'failed'
                    }
                )
            except Exception as db_error:
                logger.error(f"Failed to update transcript status: {db_error}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('extract_video_transcript', self.request.id, [url_request_id], {}, e)
        
        # Return empty result but don't break the chain (graceful degradation)
        return {'transcript_segments': 0, 'segments_created': 0, 'video_title': 'Unknown', 'error': str(e)} 