from django.db import models
from api.models import URLRequestTable

class VideoMetadata(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    url_request = models.OneToOneField(URLRequestTable, on_delete=models.CASCADE, related_name='video_metadata')
    video_id = models.CharField(max_length=20, unique=True, db_index=True, null=True, blank=True, help_text="YouTube video ID (e.g., 'dQw4w9WgXcQ')")
    title = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)
    duration = models.IntegerField(null=True, blank=True)  # in seconds
    channel_name = models.CharField(max_length=200, blank=True)
    view_count = models.BigIntegerField(null=True, blank=True)
    
    # New fields from YouTube API
    upload_date = models.DateField(null=True, blank=True, help_text="When video was published")
    language = models.CharField(max_length=10, null=True, blank=True, default='en', help_text="Primary language of the video")
    like_count = models.BigIntegerField(null=True, blank=True, help_text="Number of likes")
    channel_id = models.CharField(max_length=50, blank=True, help_text="YouTube channel ID")
    tags = models.JSONField(default=list, blank=True, help_text="Video tags for categorization")
    categories = models.JSONField(default=list, blank=True, help_text="YouTube categories")
    thumbnail = models.URLField(blank=True, help_text="Video thumbnail URL")
    channel_follower_count = models.BigIntegerField(null=True, blank=True, help_text="Channel subscriber count")
    channel_is_verified = models.BooleanField(default=False, help_text="Whether channel is verified")
    uploader_id = models.CharField(max_length=100, blank=True, help_text="Channel handle (e.g., '@TEDx')")
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)
    
    @property
    def duration_string(self):
        """Return human-readable duration (MM:SS format)"""
        if not self.duration:
            return "0:00"
        minutes = self.duration // 60
        seconds = self.duration % 60
        return f"{minutes}:{seconds:02d}"
    
    @property
    def webpage_url(self):
        """Return direct YouTube URL"""
        if self.video_id:
            return f"https://www.youtube.com/watch?v={self.video_id}"
        return ""

class VideoTranscript(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    # New relationship (will replace url_request after migration)
    video_metadata = models.OneToOneField(VideoMetadata, on_delete=models.CASCADE, related_name='video_transcript', null=True, blank=True)
    # Old relationship (keep temporarily for migration)
    url_request = models.OneToOneField(URLRequestTable, on_delete=models.CASCADE, related_name='video_transcript_old', null=True, blank=True)
    transcript_text = models.TextField()  # Plain text for search/summary - only field needed for summaries
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_formatted_transcript(self):
        """Return transcript with clickable timestamps for UI using segments"""
        if not hasattr(self, 'segments') or not self.segments.exists():
            # If no segments, return a single segment with the full text
            return [{
                'timestamp': '00:00',
                'start_seconds': 0,
                'text': self.transcript_text
            }]
        
        formatted_segments = []
        for segment in self.segments.all():
            # Format timestamp as MM:SS
            minutes = int(segment.start_time // 60)
            seconds = int(segment.start_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_segments.append({
                'timestamp': timestamp,
                'start_seconds': segment.start_time,
                'text': segment.text
            })
        return formatted_segments


class TranscriptSegment(models.Model):
    """
    Individual transcript segments for vector embedding and precise video navigation.
    Each segment represents a small portion of the video with timestamps.
    """
    
    transcript = models.ForeignKey(VideoTranscript, on_delete=models.CASCADE, related_name='segments')
    segment_id = models.CharField(max_length=50, unique=True, db_index=True, 
                                 help_text="Unique segment ID format: 'video_id_segment_number' (e.g., 'dQw4w9WgXcQ_001')")
    sequence_number = models.IntegerField(help_text="Sequential order of this segment in the video")
    
    # Timestamp information
    start_time = models.FloatField(help_text="Start time in seconds")
    duration = models.FloatField(help_text="Duration of this segment in seconds")
    
    # Content
    text = models.TextField(help_text="Text content of this segment")
    
    # Vector embedding tracking
    is_embedded = models.BooleanField(default=False, help_text="Whether this segment has been embedded in vector store")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['sequence_number']
        unique_together = ['transcript', 'sequence_number']
        indexes = [
            models.Index(fields=['transcript', 'sequence_number']),
            models.Index(fields=['start_time']),
            models.Index(fields=['is_embedded']),
        ]
    
    @property
    def end_time(self):
        """Calculate end time from next segment or use start_time + duration"""
        next_segment = self.transcript.segments.filter(
            sequence_number=self.sequence_number + 1
        ).first()
        if next_segment:
            return next_segment.start_time
        return self.start_time + self.duration
    
    @property
    def pinecone_vector_id(self):
        """Use segment_id as the Pinecone vector ID"""
        return self.segment_id
    
    def __str__(self):
        return f"{self.segment_id} ({self.start_time}s-{self.end_time}s)"
    
    def get_formatted_timestamp(self):
        """Return formatted timestamp as MM:SS for UI display"""
        minutes = int(self.start_time // 60)
        seconds = int(self.start_time % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_youtube_url_with_timestamp(self):
        """Return YouTube URL with timestamp for direct video navigation"""
        video_id = self.transcript.video_metadata.video_id
        return f"https://www.youtube.com/watch?v={video_id}&t={int(self.start_time)}s"

# Helper function to update URLRequestTable status based on related models
def update_url_request_status(url_request):
    """Update URLRequestTable status based on VideoMetadata and VideoTranscript statuses"""
    try:
        metadata_status = getattr(url_request.video_metadata, 'status', None)
        # Get transcript status through the new relationship
        transcript_status = None
        if hasattr(url_request, 'video_metadata') and hasattr(url_request.video_metadata, 'video_transcript'):
            transcript_status = getattr(url_request.video_metadata.video_transcript, 'status', None)
        
        # If both exist and both are successful
        if metadata_status == 'success' and transcript_status == 'success':
            url_request.status = 'success'
        # If either has failed
        elif metadata_status == 'failed' or transcript_status == 'failed':
            url_request.status = 'failed'
        # Otherwise keep as processing
        else:
            url_request.status = 'processing'
            
        url_request.save()
        
    except (VideoMetadata.DoesNotExist, VideoTranscript.DoesNotExist):
        # If either model doesn't exist yet, keep as processing
        url_request.status = 'processing'
        url_request.save()