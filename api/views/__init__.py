"""
API Views Module

This module contains focused view functions organized by domain.
Each view file handles a specific aspect of the API functionality.

Available view modules:
- video_views: Video processing and summary retrieval
- search_views: Video question answering and search
- status_views: Real-time status streaming
"""

# Import all view functions for easy access
from .video_views import process_single_video, get_video_summary
from .search_views import ask_video_question
from .status_views import video_status_stream

__all__ = [
    'process_single_video',
    'get_video_summary', 
    'ask_video_question',
    'video_status_stream',
]