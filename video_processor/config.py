# Video Processor Configuration

# YouTube API Configuration
YOUTUBE_CONFIG = {
    'TASK_TIMEOUTS': {
        'metadata_timeout': 300,  # 5 minutes
        'transcript_timeout': 180,  # 3 minutes - Decodo is much faster
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
        'summary': {
            'max_retries': 3,
            'countdown': 45,
            'backoff': True,
            'jitter': True,
        },
        'embedding': {
            'max_retries': 2,
            'countdown': 60,
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

# Decodo Configuration
DECODO_CONFIG = {
    'API': {
        'timeout': 30,  # 30 second timeout for Decodo requests
        'max_retries': 2,
    },
    'LANGUAGE_HANDLING': {
        'default_language': 'en',
        'fallback_languages': ['en', 'es', 'fr', 'de', 'it', 'pt'],
        'auto_detect': True,  # Try to detect video language first
    }
}

# Task State Tracking
TASK_STATES = {
    'EXTRACTING_METADATA': 'extracting_metadata',
    'EXTRACTING_TRANSCRIPT': 'extracting_transcript',
    'GENERATING_SUMMARY': 'generating_summary',
    'EMBEDDING_CONTENT': 'embedding_content',
    'UPDATING_STATUS': 'updating_status',
    'COMPLETED': 'completed',
    'FAILED_PERMANENTLY': 'failed_permanently'
}