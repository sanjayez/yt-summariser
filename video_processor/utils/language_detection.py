"""
Language detection utilities using fast-langdetect library.
Provides lightweight language detection for transcript text with fallback handling.
"""

import logging

from fast_langdetect import DetectError, LangDetectConfig, LangDetector

from video_processor.config import BUSINESS_LOGIC_CONFIG, LANGUAGE_DETECTION_CONFIG

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

    config = LANGUAGE_DETECTION_CONFIG["sample_extraction"]
    text_len = len(transcript_text)

    # Extract meaningful samples that preserve word boundaries
    # Use larger samples to get coherent text for accurate language detection
    beginning_chars = max(
        200, config["beginning_chars"] * 6
    )  # 200+ chars instead of 30
    beginning_end = min(beginning_chars, len(transcript_text))

    # Find word boundary for beginning sample
    while (
        beginning_end < len(transcript_text)
        and beginning_end > 0
        and transcript_text[beginning_end] not in " \n\t"
    ):
        beginning_end += 1

    beginning = transcript_text[:beginning_end].strip()

    # For longer texts, also extract middle and end samples with word boundaries
    if text_len > beginning_chars * 2:
        # Middle sample with word boundaries
        middle_chars = max(150, config["middle_chars"] * 4)  # Larger middle sample
        middle_start = text_len // 2 - middle_chars // 2
        middle_end = middle_start + middle_chars

        # Adjust to word boundaries
        while middle_start > 0 and transcript_text[middle_start] not in " \n\t":
            middle_start -= 1
        while (
            middle_end < len(transcript_text)
            and transcript_text[middle_end] not in " \n\t"
        ):
            middle_end += 1

        middle = transcript_text[middle_start:middle_end].strip()

        # End sample with word boundaries
        end_chars = max(150, config["end_chars"] * 4)  # Larger end sample
        end_start = max(text_len - end_chars, middle_end + 10)

        # Adjust to word boundary
        while end_start > 0 and transcript_text[end_start] not in " \n\t":
            end_start -= 1

        end = transcript_text[end_start:].strip()

        # Combine samples with spaces
        combined = f"{beginning} {middle} {end}"
    else:
        combined = beginning

    return combined.strip()


def detect_transcript_language(transcript_text: str) -> tuple[bool, float]:
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
            sample_text, low_memory=LANGUAGE_DETECTION_CONFIG["use_low_memory"]
        )

        language = result.get("lang", "")
        confidence = result.get("score", 0.0)

        # Use adaptive confidence threshold based on sample quality
        threshold = BUSINESS_LOGIC_CONFIG["LANGUAGE_DETECTION"]["confidence_threshold"]

        # Lower threshold for longer, more representative samples
        if len(sample_text) > 200:
            effective_threshold = max(0.4, threshold - 0.2)
        else:
            effective_threshold = threshold

        is_english = language == "en" and confidence >= effective_threshold

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
        return LANGUAGE_DETECTION_CONFIG["default_api_language"]

    # Remove 'fallback-' prefix for API calls
    if metadata_language.startswith("fallback-"):
        return metadata_language.replace("fallback-", "")

    return metadata_language


def should_exclude_for_language(video_metadata, transcript) -> tuple[bool, str]:
    """
    Determine if video should be excluded based on language analysis.

    This function handles edge cases in language detection:
    1. Foreign metadata + foreign transcript → exclude
    2. English transcript + empty metadata → don't exclude
    3. Foreign transcript + English metadata → exclude

    Args:
        video_metadata: VideoMetadata instance with language info
        transcript: VideoTranscript instance with transcript text

    Returns:
        Tuple[bool, str]: (should_exclude, reason)
    """
    try:
        # Get metadata language
        metadata_language = getattr(video_metadata, "language", "") or ""
        metadata_is_english = metadata_language.lower() in [
            "en",
            "en-us",
            "en-gb",
            "fallback-en",
        ]

        # Get transcript text and detect language
        transcript_text = getattr(transcript, "transcript_text", "") or ""

        # If no transcript, can't make language-based exclusion decision
        if not transcript_text.strip():
            logger.warning(
                f"No transcript text available for language check: {video_metadata.video_id}"
            )
            return False, "no_transcript_for_language_check"

        # Detect transcript language
        transcript_is_english, confidence = detect_transcript_language(transcript_text)

        logger.info(
            f"Language analysis for {video_metadata.video_id}: "
            f"metadata_lang='{metadata_language}' (is_english={metadata_is_english}), "
            f"transcript_is_english={transcript_is_english} (confidence={confidence:.3f})"
        )

        # Case 1: Both metadata and transcript are non-English → exclude
        if not metadata_is_english and not transcript_is_english:
            logger.info(
                f"Excluding {video_metadata.video_id}: both metadata and transcript are non-English"
            )
            return True, "language_unsupported"

        # Case 2: English transcript but empty/missing metadata → don't exclude
        if transcript_is_english and not metadata_language.strip():
            logger.info(
                f"Keeping {video_metadata.video_id}: English transcript with empty metadata"
            )
            return False, "english_transcript_empty_metadata"

        # Case 3: Non-English transcript but English metadata → exclude (transcript is authoritative)
        if not transcript_is_english and metadata_is_english:
            logger.info(
                f"Excluding {video_metadata.video_id}: non-English transcript overrides English metadata"
            )
            return True, "language_unsupported"

        # Case 4: Both are English → don't exclude
        if metadata_is_english and transcript_is_english:
            logger.info(
                f"Keeping {video_metadata.video_id}: both metadata and transcript are English"
            )
            return False, "both_english"

        # Fallback: if we can't determine clearly, be conservative and don't exclude
        logger.warning(
            f"Inconclusive language analysis for {video_metadata.video_id}, not excluding"
        )
        return False, "inconclusive_language_analysis"

    except Exception as e:
        logger.error(
            f"Error in language exclusion check for {getattr(video_metadata, 'video_id', 'unknown')}: {e}"
        )
        # On error, be conservative and don't exclude
        return False, f"language_check_error: {str(e)}"
