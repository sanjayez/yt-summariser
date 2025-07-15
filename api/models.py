from django.db import models
import uuid

# Create your models here.

class URLRequestTable(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
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
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Request {str(self.request_id)[:8]}"
