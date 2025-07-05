# Video Processor Configuration

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

# Task State Tracking
TASK_STATES = {
    'EXTRACTING_METADATA': 'extracting_metadata',
    'EXTRACTING_TRANSCRIPT': 'extracting_transcript', 
    'UPDATING_STATUS': 'updating_status',
    'COMPLETED': 'completed',
    'FAILED_PERMANENTLY': 'failed_permanently'
}