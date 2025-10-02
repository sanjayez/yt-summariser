import uuid

from django.core.validators import RegexValidator
from django.db import models


class StatusChoices(models.TextChoices):
    """Standard status choices for all workflow models"""

    PENDING = "pending", "Pending"
    FAILED = "failed", "Failed"
    SUCCESS = "success", "Success"


class YTInsightRun(models.Model):
    """Processing job that tracks videos from a query"""

    run_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    query_request = models.ForeignKey(
        "query_processor.QueryRequest",
        on_delete=models.CASCADE,
        related_name="insight_runs",
    )
    video_ids = models.JSONField(default=list)  # List of video IDs to process
    status = models.CharField(
        max_length=20,
        default=StatusChoices.PENDING,
        choices=StatusChoices.choices,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        db_table = "yt_insight_run"

    def __str__(self):
        return f"Run {self.run_id} - {self.status}"


class BaseResult(models.Model):
    """Abstract base for result models"""

    video_id = models.CharField(
        max_length=20,
        db_index=True,
        validators=[
            RegexValidator(
                regex=r"^[A-Za-z0-9_-]{11}$", message="Invalid YouTube video ID format"
            )
        ],
    )
    status = models.CharField(
        max_length=20,
        default=StatusChoices.PENDING,
        choices=StatusChoices.choices,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True


class VideoTable(BaseResult):
    """Track individual video processing status"""

    chapters = models.JSONField(
        default=list, blank=True, help_text="List of detected chapters with timestamps"
    )

    class Meta:
        db_table = "yt_videos"

    def __str__(self):
        return f"Video {self.video_id} - {self.status}"
