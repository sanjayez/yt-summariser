# Utils package for video processor
# Import all utility functions from the main utils.py file to maintain compatibility

import importlib.util
import os

# Direct import from the utils.py file to avoid naming conflicts
utils_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils.py")
spec = importlib.util.spec_from_file_location("video_processor_utils", utils_file_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

# Import all the utility functions
timeout = utils_module.timeout
generate_idempotency_key = utils_module.generate_idempotency_key
check_task_idempotency = utils_module.check_task_idempotency
mark_task_complete = utils_module.mark_task_complete
idempotent_task = utils_module.idempotent_task
handle_dead_letter_task = utils_module.handle_dead_letter_task
atomic_with_callback = utils_module.atomic_with_callback
update_task_progress = utils_module.update_task_progress

# Import video filtering utilities
# Import metadata normalization utilities
from .metadata_normalizer import YouTubeMetadataNormalizer, normalize_youtube_metadata

# Import music classification utilities
from .music_classification import (
    classify_music_content,
    get_music_classification_summary,
    should_exclude_for_music,
)
from .video_filtering import (
    add_video_to_exclusion_table,
    extract_video_id_from_url,
    filter_excluded_videos,
    get_exclusion_statistics,
    is_video_excluded,
)

# Re-export all imported functions
__all__ = [
    "timeout",
    "generate_idempotency_key",
    "check_task_idempotency",
    "mark_task_complete",
    "idempotent_task",
    "handle_dead_letter_task",
    "atomic_with_callback",
    "update_task_progress",
    # Video filtering utilities
    "extract_video_id_from_url",
    "is_video_excluded",
    "filter_excluded_videos",
    "add_video_to_exclusion_table",
    "get_exclusion_statistics",
    # Music classification utilities
    "classify_music_content",
    "should_exclude_for_music",
    "get_music_classification_summary",
    # Metadata normalization utilities
    "normalize_youtube_metadata",
    "YouTubeMetadataNormalizer",
]
