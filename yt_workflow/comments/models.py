from django.db import models

from yt_workflow.shared.models import BaseResult


class CommentsResult(BaseResult):
    """Comments processing results"""

    raw_comments = models.JSONField(default=list, blank=True)
    processed_insights = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "Comments Result"
        verbose_name_plural = "Comments Results"
