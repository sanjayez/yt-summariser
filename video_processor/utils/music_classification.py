"""
Smart music content classification utilities.

Distinguishes between intentional music content (keep) and background music content (exclude)
for better video processing and answer quality.
"""

import logging

from video_processor.config import BUSINESS_LOGIC_CONFIG

logger = logging.getLogger(__name__)


def classify_music_content(
    tags: list[str], categories: list[str], title: str, description: str
) -> str:
    """
    Classify video content to determine if it should be excluded for music reasons.

    Uses a weight-based classification system to distinguish between:
    - Type 1 (Intentional Music): Official music videos, concerts, performances - KEEP
    - Type 2 (Background Music): ASMR, study music, background audio - EXCLUDE

    Args:
        tags: List of video tags
        categories: List of YouTube categories
        title: Video title
        description: Video description

    Returns:
        str: Classification result
            - 'intentional_music': Keep for music content (official videos, concerts)
            - 'background_music': Exclude for background music content
            - 'not_music': Regular content, no music classification needed
    """
    try:
        # Get classification config
        config = BUSINESS_LOGIC_CONFIG["MUSIC_DETECTION"]

        # Combine all text for analysis (case-insensitive)
        all_text = " ".join([title, description] + tags).lower()

        # Count primary music signals (Type 1 indicators)
        music_score = sum(
            1 for tag in config["primary_music_tags"] if tag.lower() in all_text
        )

        # Count background music signals (Type 2 indicators)
        background_score = sum(
            1
            for indicator in config["background_indicators"]
            if indicator.lower() in all_text
        )

        # Check if video is in music categories
        has_music_category = any(
            category in config["primary_music_categories"] for category in categories
        )

        # Classification logic
        music_threshold = config["music_weight_threshold"]
        background_threshold = config["background_weight_threshold"]

        logger.debug(
            f"Music classification: music_score={music_score}, background_score={background_score}, "
            f"has_music_category={has_music_category}, title='{title[:50]}...'"
        )

        # Type 2: Background music content (EXCLUDE) - Check first to prioritize background indicators
        if background_score >= background_threshold:
            logger.info(
                f"Classified as background music: {background_score} background indicators"
            )
            return "background_music"

        # Type 1: Intentional music content (KEEP) - Only if no background indicators
        elif music_score >= music_threshold and has_music_category:
            logger.info(
                f"Classified as intentional music: {music_score} music indicators, music category"
            )
            return "intentional_music"

        # Regular content
        else:
            logger.debug(
                "Classified as regular content: insufficient music or background indicators"
            )
            return "not_music"

    except Exception as e:
        logger.error(f"Error in music classification: {e}")
        return "not_music"  # Safe fallback


def should_exclude_for_music(video_metadata) -> tuple[bool, str]:
    """
    Determine if a video should be excluded based on music classification.

    Args:
        video_metadata: VideoMetadata instance with tags, categories, title, description

    Returns:
        Tuple[bool, str]: (should_exclude, exclusion_reason)
            - should_exclude: True if video should be excluded
            - exclusion_reason: 'background_music_only' if excluded, None if not
    """
    try:
        # Extract data from video metadata (using normalized metadata - no need for getattr fallbacks)
        tags = video_metadata.tags
        categories = video_metadata.categories
        title = video_metadata.title
        description = video_metadata.description

        # Classify the content
        classification = classify_music_content(tags, categories, title, description)

        # Only exclude background music content
        if classification == "background_music":
            logger.info(
                f"Video {video_metadata.video_id} will be excluded: background music content"
            )
            return True, "background_music_only"
        else:
            logger.debug(
                f"Video {video_metadata.video_id} classification: {classification} (not excluded)"
            )
            return False, None

    except Exception as e:
        logger.error(
            f"Error checking music exclusion for video {video_metadata.video_id}: {e}"
        )
        return False, None  # Safe fallback - don't exclude on error


def get_music_classification_summary(
    tags: list[str], categories: list[str], title: str, description: str
) -> dict:
    """
    Get detailed music classification analysis for debugging/analytics.

    Args:
        tags: List of video tags
        categories: List of YouTube categories
        title: Video title
        description: Video description

    Returns:
        dict: Detailed classification analysis
    """
    try:
        config = BUSINESS_LOGIC_CONFIG["MUSIC_DETECTION"]
        all_text = " ".join([title, description] + tags).lower()

        # Find matching indicators
        matching_music_tags = [
            tag for tag in config["primary_music_tags"] if tag.lower() in all_text
        ]

        matching_background_indicators = [
            indicator
            for indicator in config["background_indicators"]
            if indicator.lower() in all_text
        ]

        has_music_category = any(
            category in config["primary_music_categories"] for category in categories
        )

        classification = classify_music_content(tags, categories, title, description)

        return {
            "classification": classification,
            "music_score": len(matching_music_tags),
            "background_score": len(matching_background_indicators),
            "has_music_category": has_music_category,
            "matching_music_tags": matching_music_tags,
            "matching_background_indicators": matching_background_indicators,
            "thresholds": {
                "music_threshold": config["music_weight_threshold"],
                "background_threshold": config["background_weight_threshold"],
            },
        }

    except Exception as e:
        logger.error(f"Error generating music classification summary: {e}")
        return {"classification": "error", "error": str(e)}
