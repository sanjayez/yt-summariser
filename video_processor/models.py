from django.db import models
from api.models import URLRequestTable

class VideoMetadata(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    url_request = models.OneToOneField(URLRequestTable, on_delete=models.CASCADE, related_name='video_metadata')
    title = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)
    duration = models.IntegerField(null=True, blank=True)  # in seconds
    channel_name = models.CharField(max_length=200, blank=True)
    view_count = models.BigIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)

class VideoTranscript(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    url_request = models.OneToOneField(URLRequestTable, on_delete=models.CASCADE, related_name='video_transcript')
    transcript_text = models.TextField()
    language = models.CharField(max_length=10, default='en')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)

# Helper function to update URLRequestTable status based on related models
def update_url_request_status(url_request):
    """Update URLRequestTable status based on VideoMetadata and VideoTranscript statuses"""
    try:
        metadata_status = getattr(url_request.video_metadata, 'status', None)
        transcript_status = getattr(url_request.video_transcript, 'status', None)
        
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