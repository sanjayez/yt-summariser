"""
Top-level task imports for Celery autodiscovery.

This module exposes all tasks from nested modules so Celery can discover them automatically.
Import this module in your Celery configuration or let Celery autodiscover it.
"""

# Import all tasks to make them discoverable by Celery
from yt_workflow.comments.tasks import process_comments
from yt_workflow.metadata.tasks import process_metadata
from yt_workflow.orchestrator import yt_orchestrator
from yt_workflow.shared.tasks.aggregate_results import aggregate_results
from yt_workflow.shared.tasks.pipeline import yt_pipeline
from yt_workflow.transcript.tasks import process_transcript

# Expose tasks for explicit imports
__all__ = [
    "yt_orchestrator",
    "yt_pipeline",
    "aggregate_results",
    "process_metadata",
    "process_transcript",
    "process_comments",
]
