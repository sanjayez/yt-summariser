import uuid
from django.db import models


class SearchSession(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user_ip = models.GenericIPAddressField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Session {str(self.session_id)[:8]}"
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['user_ip']),
            models.Index(fields=['created_at']),
        ]


class SearchRequest(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    INTENT_CHOICES = [
        ('LOOKUP', 'Lookup'),
        ('TUTORIAL', 'Tutorial'),
        ('HOW_TO', 'How To'),
        ('REVIEW', 'Review'),
    ]
    
    search_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    search_session = models.ForeignKey(SearchSession, on_delete=models.CASCADE, related_name='search_requests')
    original_query = models.TextField(help_text="User's original query")
    processed_query = models.TextField(blank=True, help_text="LLM-enhanced query for YouTube search")
    intent_type = models.CharField(
        max_length=20, 
        choices=INTENT_CHOICES, 
        blank=True,
        help_text="Classified intent of the user query"
    )
    video_urls = models.JSONField(default=list, blank=True, help_text="List of YouTube video URLs found")
    total_videos = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    error_message = models.TextField(blank=True, help_text="Error details if status is failed")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Search {str(self.search_id)[:8]} - {self.original_query[:50]}"
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['search_id']),
            models.Index(fields=['search_session', 'created_at']),
            models.Index(fields=['status']),
        ]