from django.db import models

from yt_workflow.shared.models import BaseResult


class TranscriptResult(BaseResult):
    """Transcript processing results"""

    raw_transcript = models.TextField(blank=True)
    processed_segments = models.JSONField(default=list, blank=True)

    class Meta:
        verbose_name = "Transcript Result"
        verbose_name_plural = "Transcript Results"
