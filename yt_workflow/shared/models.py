import uuid

from django.db import models


class WorkflowExecution(models.Model):
    """Main workflow tracking model"""

    execution_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    video_url = models.URLField(max_length=500)
    video_id = models.CharField(max_length=20, db_index=True)
    status = models.CharField(max_length=20, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Workflow {self.video_id} - {self.status}"


class BaseResult(models.Model):
    """Abstract base for result models"""

    execution = models.OneToOneField(
        WorkflowExecution, on_delete=models.CASCADE, related_name="%(class)s_result"
    )
    status = models.CharField(max_length=20, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True
