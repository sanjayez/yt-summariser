from django.db import models
import uuid

# Create your models here.

class URLRequestTable(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('success', 'Success'),
        ('failed', 'Failed'),
    ]
    
    FAILURE_REASON_CHOICES = [
        ('excluded', 'Excluded'),
        ('no_transcript', 'No Transcript'),
        ('no_metadata', 'No Metadata'),
        ('technical_failure', 'Technical Failure'),
    ]
    
    request_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    search_request = models.ForeignKey(
        'topic.SearchRequest', 
        on_delete=models.CASCADE,
        related_name='url_requests',
        null=True,
        blank=True,
        help_text="Link to the search request that generated this URL"
    )
    url = models.URLField(max_length=500)
    ip_address = models.GenericIPAddressField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    celery_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="ID of the main Celery task processing this video"
    )
    chain_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="ID of the Celery chain task for result tracking"
    )
    failure_reason = models.CharField(
        max_length=20,
        choices=FAILURE_REASON_CHOICES,
        null=True,
        blank=True,
        help_text="Reason why the video processing failed"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Request {str(self.request_id)[:8]}"
    
    class Meta:
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['failure_reason']),
            models.Index(fields=['created_at']),
            models.Index(fields=['status', 'created_at']),  # Composite index for common queries
            models.Index(fields=['status', 'failure_reason']),  # Composite index for filtering
        ]
        ordering = ['-created_at']
