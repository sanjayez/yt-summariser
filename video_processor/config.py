# Video Processor Configuration

# YouTube API Configuration
YOUTUBE_CONFIG = {
    "TASK_TIMEOUTS": {
        # Metadata extraction timeouts
        "metadata_soft_limit": 240,  # 4 minutes - gives cleanup time
        "metadata_hard_limit": 300,  # 5 minutes - absolute maximum
        "metadata_timeout": 210,  # 3.5 minutes - internal timeout
        # Transcript extraction timeouts
        "transcript_soft_limit": 180,  # 3 minutes - transcript extraction can be slow
        "transcript_hard_limit": 240,  # 4 minutes - absolute maximum
        "transcript_timeout": 150,  # 2.5 minutes - internal timeout
        # Summary generation timeouts
        "summary_soft_limit": 120,  # 2 minutes - LLM calls are usually fast
        "summary_hard_limit": 180,  # 3 minutes - absolute maximum
        "summary_timeout": 90,  # 1.5 minutes - internal timeout
        # Embedding generation timeouts
        "embedding_soft_limit": 120,  # 2 minutes - OpenAI API is usually fast
        "embedding_hard_limit": 180,  # 3 minutes - absolute maximum
        "embedding_timeout": 90,  # 1.5 minutes - internal timeout
        # Status update timeouts - these should be very fast
        "status_soft_limit": 25,  # 25 seconds - database operations
        "status_hard_limit": 30,  # 30 seconds - absolute maximum
        "status_timeout": 20,  # 20 seconds - internal timeout
        # Workflow orchestration timeouts
        "workflow_soft_limit": 600,  # 10 minutes - manages entire pipeline
        "workflow_hard_limit": 720,  # 12 minutes - absolute maximum
        "workflow_timeout": 540,  # 9 minutes - internal timeout
        # Search processing timeouts
        "search_soft_limit": 180,  # 3 minutes - YouTube search + LLM processing
        "search_hard_limit": 240,  # 4 minutes - absolute maximum
        "search_timeout": 150,  # 2.5 minutes - internal timeout
        # Parallel processing timeouts
        "parallel_soft_limit": 1200,  # 20 minutes - processing multiple videos
        "parallel_hard_limit": 1500,  # 25 minutes - absolute maximum
        "parallel_timeout": 1080,  # 18 minutes - internal timeout
        # Content analysis timeouts (fire-and-forget)
        "content_analysis_soft_limit": 600,  # 10 minutes - LLM chunked analysis
        "content_analysis_hard_limit": 900,  # 15 minutes - absolute maximum
        "content_analysis_timeout": 540,  # 9 minutes - internal timeout
    },
    "RETRY_CONFIG": {
        "metadata": {
            "max_retries": 3,
            "countdown": 60,
            "backoff": True,
            "jitter": True,
        },
        "transcript": {
            "max_retries": 2,
            "countdown": 30,
            "backoff": True,
            "jitter": True,
        },
        "summary": {
            "max_retries": 3,
            "countdown": 45,
            "backoff": True,
            "jitter": True,
        },
        "embedding": {
            "max_retries": 2,
            "countdown": 60,
            "backoff": True,
            "jitter": True,
        },
        "status_update": {
            "max_retries": 5,
            "countdown": 5,
            "backoff": False,
            "jitter": False,
        },
        "search": {
            "max_retries": 2,
            "countdown": 60,
            "backoff": True,
            "jitter": True,
        },
        "parallel": {
            "max_retries": 1,
            "countdown": 300,
            "backoff": False,
            "jitter": False,
        },
    },
}

# Decodo Configuration
DECODO_CONFIG = {
    "API": {
        "timeout": 30,  # 30 second timeout for Decodo requests
        "max_retries": 2,
    },
    "LANGUAGE_HANDLING": {
        "default_language": "en",
        "fallback_languages": ["en", "es", "fr", "de", "it", "pt"],
        "auto_detect": True,  # Try to detect video language first
    },
}

# Language Detection Configuration
LANGUAGE_DETECTION_CONFIG = {
    "confidence_threshold": 0.7,  # 70% confidence for English detection
    "fallback_language_code": "fallback-en",
    "default_api_language": "en",
    "sample_extraction": {
        "beginning_chars": 30,  # Small sample from beginning
        "middle_chars": 25,  # Small sample from middle
        "end_chars": 30,  # Small sample from end
    },
    "use_low_memory": True,  # Use low memory mode for lightweight operation
}

# Transcript Extraction Configuration
TRANSCRIPT_CONFIG = {
    "DECODO": {
        "enabled": True,
        "timeout": 30,
        "max_retries": 2,
        "priority": 1,  # Try first
    },
    "YOUTUBE_API": {
        "enabled": True,
        "timeout": 20,
        "max_retries": 1,
        "priority": 2,  # Fallback
        "preferred_languages": ["en", "es", "fr", "de"],
    },
    "FALLBACK_STRATEGY": {
        "enable_fallback": True,
        "fail_fast": False,  # Try all methods before failing
        "log_attempts": True,
    },
}

# API and SSE Configuration
API_CONFIG = {
    "POLLING": {
        "status_check_interval": 2,  # Seconds between status checks
        "status_check_max_attempts": 60,  # Max attempts for status polling (60 * 2 = 120 seconds)
        "search_polling_interval": 2,  # Seconds between search status checks
        "task_polling_interval": 1,  # Seconds between task status checks
    },
    "SSE": {
        "keepalive_interval": 30,  # Seconds between keepalive messages
        "event_timeout": 120,  # Max seconds for SSE connection
    },
    "RATE_LIMITS": {
        "default_requests_per_minute": 60,
        "burst_requests": 100,
        "ip_based": True,
    },
}

# Task State Tracking
TASK_STATES = {
    "EXTRACTING_METADATA": "extracting_metadata",
    "EXTRACTING_TRANSCRIPT": "extracting_transcript",
    "GENERATING_SUMMARY": "generating_summary",
    "EMBEDDING_CONTENT": "embedding_content",
    "UPDATING_STATUS": "updating_status",
    "ANALYZING_CONTENT": "analyzing_content",
    "COMPLETED": "completed",
    "FAILED_PERMANENTLY": "failed_permanently",
}

# Business Logic Configuration
BUSINESS_LOGIC_CONFIG = {
    "LANGUAGE_DETECTION": {
        "confidence_threshold": 0.7,  # 70% for English detection
    },
    "PIPELINE_SUCCESS": {
        "minimum_threshold": 0.85,  # 85% for URLRequest success
    },
    "DURATION_LIMITS": {
        "minimum_seconds": 60,  # 1 minute minimum
        "maximum_seconds": 900,  # 15 minutes (eventually remove)
    },
    "MUSIC_DETECTION": {
        # Comprehensive list of primary music indicators (Type 1 - KEEP)
        "primary_music_tags": [
            "music",
            "song",
            "artist",
            "album",
            "band",
            "singer",
            "musician",
            "concert",
            "performance",
            "live",
            "acoustic",
            "cover",
            "remix",
            "official",
            "mv",
            "music video",
            "single",
            "ep",
            "soundtrack",
            "composer",
            "orchestra",
            "symphony",
            "piano",
            "guitar",
            "violin",
            "jazz",
            "rock",
            "pop",
            "classical",
            "hip hop",
            "rap",
            "country",
            "folk",
            "blues",
            "metal",
            "electronic",
            "dance",
            "house",
            "techno",
        ],
        # Background music indicators (Type 2 - EXCLUDE)
        # Single array - ANY of these suggests background music content
        "background_indicators": [
            # ASMR related
            "asmr",
            "triggers",
            "tingles",
            "relaxing",
            "sleep",
            "calming",
            # Study/productivity related (specific to background music context)
            "study music",
            "focus music",
            "concentration music",
            "productivity music",
            "coding music",
            "programming music",
            "reading music",
            "homework music",
            "exam music",
            "meditation",
            # Ambient/background related
            "ambient",
            "background",
            "atmospheric",
            "soundscape",
            "white noise",
            "brown noise",
            "pink noise",
            "binaural",
            "beats for",
            "music for",
            # Nature/environmental sounds
            "rain",
            "ocean",
            "forest",
            "birds",
            "nature sounds",
            "thunderstorm",
            "waves",
            "wind",
            "fire crackling",
            "cafe sounds",
            "library",
            # Activity-specific background audio (be more specific to avoid false positives)
            "yoga",
            "massage",
            "spa",
            "workout",
            "gym",
            "running",
            "walking",
            "driving",
            "cleaning",
            "gaming",
            "streaming",
            # Explicit background music indicators
            "background music",
            "bgm",
            "instrumental only",
            "no vocals",
            "non copyright",
            "royalty free",
            "copyright free",
            "cc0",
            # Time/duration specific (often background)
            "hours",
            "hour",
            "24/7",
            "loop",
            "repeat",
            "extended",
            "long",
            "8 hours",
            "10 hours",
            "12 hours",
            "all night",
            "marathon",
        ],
        # YouTube categories that indicate primary music content
        "primary_music_categories": ["Music", "Entertainment"],
        # Weight thresholds for classification
        "music_weight_threshold": 2,  # Need 2+ primary music indicators
        "background_weight_threshold": 1,  # Need 1+ background indicator
    },
}
