"""
YouTube Transcript Service - Fallback transcript extraction
"""

import logging
from typing import Tuple, Dict, Any, List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

logger = logging.getLogger(__name__)

def extract_youtube_transcript_fallback(video_id: str, preferred_languages: List[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Extract transcript using YouTube's native transcript API as fallback.
    
    Args:
        video_id: YouTube video ID
        preferred_languages: List of preferred language codes (default: ['en'])
        
    Returns:
        Tuple of (success: bool, result: dict)
    """
    if not preferred_languages:
        preferred_languages = ['en']
    
    logger.info(f"Attempting YouTube transcript extraction for {video_id}")
    
    try:
        # List available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        available_transcripts = []
        for transcript in transcript_list:
            available_transcripts.append({
                'language_code': transcript.language_code,
                'language': transcript.language,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            })
        
        logger.info(f"Found {len(available_transcripts)} available transcripts for {video_id}")
        
        # Try to get transcript in preferred languages
        transcript_data = None
        used_language = None
        
        for lang in preferred_languages:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                used_language = lang
                logger.info(f"Successfully extracted transcript in {lang} for {video_id}")
                break
            except (NoTranscriptFound, Exception) as e:
                logger.debug(f"No transcript found in {lang} for {video_id}: {e}")
                continue
        
        # If preferred languages failed, try any available transcript
        if not transcript_data and available_transcripts:
            for transcript_info in available_transcripts:
                try:
                    lang = transcript_info['language_code']
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    used_language = lang
                    logger.info(f"Extracted transcript in fallback language {lang} for {video_id}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to extract transcript in {lang}: {e}")
                    continue
        
        if not transcript_data:
            return False, {
                'success': False,
                'error': 'No transcripts could be extracted',
                'available_transcripts': available_transcripts,
                'transcript_text': '',
                'segments': []
            }
        
        # Convert YouTube format to our standard format
        segments = []
        transcript_text_parts = []
        
        for item in transcript_data:
            segment = {
                'start': item.get('start', 0),
                'duration': item.get('duration', 0),
                'text': item.get('text', '').strip()
            }
            
            if segment['text']:
                segments.append(segment)
                transcript_text_parts.append(segment['text'])
        
        transcript_text = ' '.join(transcript_text_parts)
        
        return True, {
            'success': True,
            'transcript_text': transcript_text,
            'segments': segments,
            'language': used_language,
            'source': 'youtube_api',
            'segment_count': len(segments),
            'available_transcripts': available_transcripts
        }
        
    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video {video_id}")
        return False, {
            'success': False,
            'error': 'Transcripts are disabled for this video',
            'transcript_text': '',
            'segments': []
        }
        
    except Exception as e:
        logger.error(f"YouTube transcript extraction failed for {video_id}: {e}")
        return False, {
            'success': False,
            'error': f'YouTube transcript extraction failed: {str(e)}',
            'transcript_text': '',
            'segments': []
        } 