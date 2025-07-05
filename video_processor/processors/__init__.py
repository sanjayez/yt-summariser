# Import all processors for easy access
from .metadata import extract_video_metadata
from .transcript import extract_video_transcript
from .summary import generate_video_summary
from .embedding import embed_video_content
from .status import update_overall_status
from .workflow import process_youtube_video

__all__ = [
    'process_youtube_video',
    'extract_video_metadata', 
    'extract_video_transcript',
    'generate_video_summary',
    'embed_video_content',
    'update_overall_status'
] 