"""
Query Processor Models
Enhanced QueryRequest model supporting video, playlist, and topic requests
"""

import uuid
from django.db import models


class QueryRequest(models.Model):
    """
    Enhanced QueryRequest model that handles all request types:
    - video: Single YouTube video processing
    - playlist: YouTube playlist processing  
    - topic: Topic search with LLM enhancement
    
    Integrates with UnifiedSession for consistent rate limiting and session management.
    """
    
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('success', 'Success'),
        ('failed', 'Failed'),
    ]
    
    REQUEST_TYPE_CHOICES = [
        ('video', 'Single Video'),
        ('playlist', 'Playlist'),
        ('topic', 'Topic Search'),
    ]
    
    INTENT_CHOICES = [
        ('LOOKUP', 'Lookup'),
        ('TUTORIAL', 'Tutorial'),
        ('HOW_TO', 'How To'),
        ('REVIEW', 'Review'),
    ]
    
    # Core identification
    search_id = models.UUIDField(
        default=uuid.uuid4, 
        editable=False, 
        unique=True,
        help_text="Unique identifier for this search request"
    )
    
    unified_session = models.ForeignKey(
        'api.UnifiedSession', 
        on_delete=models.CASCADE, 
        related_name='search_requests',
        help_text="Link to UnifiedSession for rate limiting and session management"
    )
    
    # Request details
    request_type = models.CharField(
        max_length=20, 
        choices=REQUEST_TYPE_CHOICES,
        help_text="Type of request: video, playlist, or topic search"
    )
    
    original_content = models.TextField(
        help_text="Original user input: URL for video/playlist requests, query for topic requests"
    )
    
    # Topic-specific fields (nullable for video/playlist requests)
    concepts = models.JSONField(
        default=list, 
        blank=True, 
        null=True,
        help_text="LLM-extracted concepts from user query (topic requests only)"
    )
    
    enhanced_queries = models.JSONField(
        default=list, 
        blank=True, 
        null=True,
        help_text="LLM-generated enhanced queries for search execution (topic requests only)"
    )
    
    intent_type = models.CharField(
        max_length=20, 
        choices=INTENT_CHOICES, 
        blank=True, 
        null=True,
        help_text="LLM-classified intent of the user query (topic requests only)"
    )
    
    # Results (always stored as array for consistency)
    video_urls = models.JSONField(
        default=list, 
        blank=True,
        help_text="List of YouTube video URLs found/processed"
    )
    
    total_videos = models.IntegerField(
        default=0,
        help_text="Total number of videos found/processed"
    )
    
    # Status tracking
    status = models.CharField(
        max_length=20, 
        choices=STATUS_CHOICES, 
        default='processing',
        help_text="Current processing status"
    )
    
    error_message = models.TextField(
        blank=True,
        help_text="Error details if status is failed"
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this search request was created"
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['search_id']),
            models.Index(fields=['unified_session', 'created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['request_type']),
            models.Index(fields=['created_at']),
        ]
        verbose_name = "Search Request"
        verbose_name_plural = "Search Requests"
    
    def __str__(self):
        content_preview = self.original_content[:50] + "..." if len(self.original_content) > 50 else self.original_content
        return f"Search {str(self.search_id)[:8]} - {self.request_type} - {content_preview}"
    
    def is_topic_request(self):
        """Check if this is a topic search request"""
        return self.request_type == 'topic'
    
    def is_video_request(self):
        """Check if this is a video or playlist request"""
        return self.request_type in ['video', 'playlist']
    
    def get_session_ip(self):
        """Get the IP address from the associated session"""
        return self.unified_session.user_ip if self.unified_session else None
    
    def get_processing_summary(self):
        """Get a summary of processing results"""
        if self.status == 'processing':
            return f"Processing {self.request_type} request..."
        elif self.status == 'success':
            if self.is_topic_request():
                return f"Found {self.total_videos} videos for topic search"
            else:
                return f"Successfully processed {self.request_type}"
        elif self.status == 'failed':
            return f"Failed to process {self.request_type}: {self.error_message}"
        return "Unknown status"