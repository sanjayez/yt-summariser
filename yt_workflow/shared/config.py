# Configuration for yt_workflow tasks

TASK_TIMEOUTS = {
    "metadata_fetch": 240,  # 4 minutes
    "transcript_fetch": 180,  # 3 minutes
    "comments_fetch": 300,  # 5 minutes
    "processing": 600,  # 10 minutes
}

RETRY_CONFIG = {
    "max_retries": 3,
    "countdown": 60,
    "backoff": True,
    "jitter": True,
}
