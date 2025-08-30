# Backwards compatibility imports - all tasks have been moved to processors/ module
from .processors.embedding import embed_video_content
from .processors.metadata import extract_video_metadata
from .processors.status import update_overall_status
from .processors.summary import generate_video_summary
from .processors.transcript import extract_video_transcript
from .processors.workflow import process_youtube_video

# Re-export all tasks for backwards compatibility
__all__ = [
    "process_youtube_video",
    "extract_video_metadata",
    "extract_video_transcript",
    "generate_video_summary",
    "embed_video_content",
    "update_overall_status",
]
