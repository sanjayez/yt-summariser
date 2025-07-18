"""
Language detection utilities using fast-langdetect library.
Provides lightweight language detection for transcript text with fallback handling.
"""

import logging
from typing import Tuple
from fast_langdetect import detect, DetectError, LangDetectConfig, LangDetector
from video_processor.config import LANGUAGE_DETECTION_CONFIG

logger = logging.getLogger(__name__)


def extract_text_samples(transcript_text: str) -> str:
    """
    Extract small samples from beginning, middle, and end of transcript text.
    Total sample size: ~85 characters for efficient detection.
    
    Args:
        transcript_text: Full transcript text
        
    Returns:
        Combined sample text for language detection
    """
    if not transcript_text or len(transcript_text) < 10:
        return transcript_text
    
    config = LANGUAGE_DETECTION_CONFIG['sample_extraction']
    text_len = len(transcript_text)
    
    # Extract small sample from beginning
    beginning = transcript_text[:config['beginning_chars']].strip()
    
    # For longer texts, also extract middle and end samples
    if text_len > config['beginning_chars'] * 2:
        # Middle sample
        middle_start = text_len // 2 - config['middle_chars'] // 2
        middle_end = middle_start + config['middle_chars']
        middle = transcript_text[middle_start:middle_end].strip()
        
        # End sample
        end_start = max(text_len - config['end_chars'], middle_end + 10)
        end = transcript_text[end_start:].strip()
        
        # Combine samples with spaces
        combined = f"{beginning} {middle} {end}"
    else:
        combined = beginning
    
    return combined.strip()


def detect_transcript_language(transcript_text: str) -> Tuple[bool, float]:
    """
    Detect if transcript text is English with confidence score.
    Uses fast-langdetect with low memory mode for lightweight operation.
    
    Args:
        transcript_text: Full transcript text
        
    Returns:
        Tuple of (is_english, confidence_score)
        - is_english: True if detected as English with >70% confidence
        - confidence_score: Float between 0-1
    """
    try:
        # Extract small samples for detection
        sample_text = extract_text_samples(transcript_text)
        
        if not sample_text or len(sample_text) < 5:
            logger.warning("Text too short for language detection")
            return False, 0.0
        
        # Configure detector with fallback support
        config = LangDetectConfig(
            cache_dir=None,  # Use default temp dir
            allow_fallback=True,  # Enable fallback to small model if needed
        )
        detector = LangDetector(config)
        
        # Detect language using fast-langdetect with low memory mode
        result = detector.detect(
            sample_text, 
            low_memory=LANGUAGE_DETECTION_CONFIG['use_low_memory']
        )
        
        language = result.get('lang', '')
        confidence = result.get('score', 0.0)
        
        # Check if English and meets confidence threshold
        is_english = (
            language == 'en' and 
            confidence >= LANGUAGE_DETECTION_CONFIG['confidence_threshold']
        )
        
        logger.info(
            f"Language detection: lang={language}, confidence={confidence:.3f}, "
            f"is_english={is_english}, sample_chars={len(sample_text)}"
        )
        
        return is_english, confidence
        
    except DetectError as e:
        logger.warning(f"Language detection failed: {e}")
        return False, 0.0
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}")
        return False, 0.0


def should_update_language(current_language: str) -> bool:
    """
    Check if language should be updated.
    Only processes fallback-* languages to preserve audit trail.
    
    Args:
        current_language: Current language code from metadata
        
    Returns:
        True if language should be processed for detection and update
    """
    return current_language and current_language.startswith('fallback-')


def get_api_language(metadata_language: str) -> str:
    """
    Convert metadata language to API-compatible language code.
    Converts 'fallback-en' -> 'en' for API calls.
    
    Args:
        metadata_language: Language from VideoMetadata
        
    Returns:
        API-compatible language code
    """
    if not metadata_language:
        return LANGUAGE_DETECTION_CONFIG['default_api_language']
    
    # Remove 'fallback-' prefix for API calls
    if metadata_language.startswith('fallback-'):
        return metadata_language.replace('fallback-', '')
    
    return metadata_language 