# Backwards compatibility imports - all tasks have been moved to processors/ module
from .processors.workflow import process_youtube_video
from .processors.metadata import extract_video_metadata
from .processors.transcript import extract_video_transcript
from .processors.status import update_overall_status

# Re-export all tasks for backwards compatibility
__all__ = [
    'process_youtube_video',
    'extract_video_metadata',
    'extract_video_transcript', 
    'update_overall_status'
]