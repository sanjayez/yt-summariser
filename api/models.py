from django.db import models
from django.db.models import F
from django.conf import settings
from django.db.models.functions import Extract
from django.utils import timezone
import uuid

# Create your models here.


class UnifiedSession(models.Model):
    """Unified session tracking for all request types (video, playlist, topic)"""
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_ip = models.GenericIPAddressField()
    
    # Request counters for rate limiting
    video_requests = models.PositiveSmallIntegerField(default=0, help_text="Number of single video processing requests")
    playlist_requests = models.PositiveSmallIntegerField(default=0, help_text="Number of playlist processing requests")
    topic_requests = models.PositiveSmallIntegerField(default=0, help_text="Number of topic search requests")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    last_request_at = models.DateTimeField(auto_now=True)
    
    # Future account integration (for post-alpha account-based tracking)
    user_account = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        null=True, 
        blank=True,
        on_delete=models.SET_NULL,
        help_text="Associated user account (null for anonymous sessions)"
    )
    
    class Meta:
        ordering = ['-last_request_at']
        indexes = [
            models.Index(fields=['user_ip']),
            models.Index(fields=['created_at']),
            models.Index(fields=['last_request_at']),
        ]
        constraints = [
            # Race condition prevention handled at application level via SessionService
        ]
    
    def __str__(self):
        return f"Session {str(self.session_id)[:8]} - {self.user_ip}"
    
    @property
    def total_requests(self):
        """Calculate total requests across all types"""
        return self.video_requests + self.playlist_requests + self.topic_requests
    
    def can_make_request(self):
        """Check if user can make another request (3 per day limit for alpha)"""
        return self.total_requests < 3
    
    def increment_request_count(self, request_type):
        """Increment counter for specific request type and update last_request_at (atomic)."""
        field_map = {
            'video': 'video_requests',
            'playlist': 'playlist_requests',
            'topic': 'topic_requests'
        }
        field = field_map.get(request_type)
        if not field:
            raise ValueError(f"Invalid request type: {request_type}")
        
        # Atomic database update
        updates = {
            field: F(field) + 1,
            'last_request_at': timezone.now()
        }
        UnifiedSession.objects.filter(pk=self.pk).update(**updates)
        
        # Refresh instance to reflect DB changes
        self.refresh_from_db(fields=[
            'video_requests', 'playlist_requests', 'topic_requests', 'last_request_at'
        ])
    
    def get_remaining_requests(self):
        """Get number of remaining requests for the day"""
        return max(0, 3 - self.total_requests)

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
    
    request_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
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
