from django.core.validators import RegexValidator
from django.db import models


class TranscriptSegment(models.Model):
    """Individual transcript lines with normalized timestamps and content"""

    line_id = models.CharField(
        max_length=50,
        primary_key=True,
        validators=[
            RegexValidator(
                regex=r"^[A-Za-z0-9_-]{11}_line_\d+$",
                message="Invalid line_id format. Expected: video_id_line_idx",
            )
        ],
        help_text="Format: video_id_line_idx (e.g., t1hOdm0RJlY_line_42)",
    )
    video_id = models.CharField(
        max_length=11,
        db_index=True,
        validators=[
            RegexValidator(
                regex=r"^[A-Za-z0-9_-]{11}$", message="Invalid YouTube video ID format"
            )
        ],
    )
    idx = models.PositiveIntegerField(help_text="1-based line index within the video")
    start = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="Start time in seconds"
    )
    end = models.DecimalField(
        max_digits=10, decimal_places=2, help_text="End time in seconds"
    )
    text = models.TextField(help_text="Normalized transcript text content")

    class Meta:
        verbose_name = "Transcript Segment"
        verbose_name_plural = "Transcript Segments"
        db_table = "transcript"
        indexes = [
            models.Index(fields=["video_id", "idx"]),
            models.Index(fields=["video_id", "start"]),
            models.Index(fields=["start", "end"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(start__gte=0), name="transcript_start_non_negative"
            ),
            models.CheckConstraint(
                check=models.Q(end__gt=models.F("start")),
                name="transcript_end_after_start",
            ),
            models.CheckConstraint(
                check=models.Q(idx__gt=0), name="transcript_idx_positive"
            ),
        ]
        ordering = ["video_id", "idx"]

    def __str__(self):
        return f"{self.line_id}: {self.start}s-{self.end}s"

    @property
    def duration(self):
        """Calculate line duration in seconds"""
        return float(self.end - self.start)
