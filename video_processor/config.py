# Video Processor Configuration
import re
from urllib.parse import urlparse

# YouTube API Configuration
YOUTUBE_CONFIG = {
    'YDL_OPTS': {
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 2,
        'extract_flat': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
    },
    'TASK_TIMEOUTS': {
        'metadata_timeout': 300,  # 5 minutes
        'transcript_timeout': 180,  # 3 minutes
        'status_update_timeout': 30,  # 30 seconds
    },
    'RETRY_CONFIG': {
        'metadata': {
            'max_retries': 3,
            'countdown': 60,
            'backoff': True,
            'jitter': True,
        },
        'transcript': {
            'max_retries': 2,
            'countdown': 30,
            'backoff': True,
            'jitter': True,
        },
        'status_update': {
            'max_retries': 5,
            'countdown': 5,
            'backoff': False,
            'jitter': False,
        }
    }
}

# URL Validation
YOUTUBE_URL_REGEX = re.compile(
    r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
)

# Task State Tracking
TASK_STATES = {
    'EXTRACTING_METADATA': 'extracting_metadata',
    'EXTRACTING_TRANSCRIPT': 'extracting_transcript', 
    'UPDATING_STATUS': 'updating_status',
    'COMPLETED': 'completed',
    'FAILED_PERMANENTLY': 'failed_permanently'
}

# Validation Functions
def validate_youtube_url(url):
    """Validate YouTube URL format"""
    if not url:
        raise ValueError("URL cannot be empty")
    
    if not YOUTUBE_URL_REGEX.match(url):
        raise ValueError("Invalid YouTube URL format")
    
    return url

def validate_video_info(info):
    """Validate video information from yt-dlp"""
    if not info:
        raise ValueError("No video information extracted")
    
    required_fields = ['id', 'title']
    missing_fields = [field for field in required_fields if not info.get(field)]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Additional validation
    if len(info.get('id', '')) != 11:
        raise ValueError("Invalid video ID format")
    
    return info

def validate_transcript_data(transcript_data):
    """Validate transcript data structure"""
    if not transcript_data:
        raise ValueError("Transcript data is empty")
    
    if not isinstance(transcript_data, list):
        raise ValueError("Transcript data must be a list")
    
    # Validate first segment structure
    if transcript_data and isinstance(transcript_data[0], dict):
        required_keys = ['text', 'start']
        first_segment = transcript_data[0]
        missing_keys = [key for key in required_keys if key not in first_segment]
        
        if missing_keys:
            raise ValueError(f"Transcript segments missing required keys: {missing_keys}")
    
    return transcript_data 