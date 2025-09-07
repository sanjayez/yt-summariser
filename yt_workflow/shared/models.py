import uuid

from django.db import models


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
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("failed", "Failed"),
            ("success", "Success"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        db_table = "yt_insight_run"

    def __str__(self):
        return f"Run {self.run_id} - {self.status}"


class BaseResult(models.Model):
    """Abstract base for result models"""

    video_id = models.CharField(max_length=20, db_index=True)
    status = models.CharField(
        max_length=20,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("failed", "Failed"),
            ("success", "Success"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True


class VideoTable(BaseResult):
    """Track individual video processing status"""

    class Meta:
        db_table = "video_table"

    def __str__(self):
        return f"Video {self.video_id} - {self.status}"
