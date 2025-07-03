from django.contrib import admin

# Register your models here.
from .models import VideoMetadata, VideoTranscript

@admin.register(VideoMetadata)
class VideoMetadataAdmin(admin.ModelAdmin):
    list_display = ['title', 'channel_name', 'duration', 'view_count', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['title', 'channel_name']
    readonly_fields = ['created_at']

@admin.register(VideoTranscript)
class VideoTranscriptAdmin(admin.ModelAdmin):
    list_display = ['url_request', 'language', 'status', 'transcript_preview', 'created_at']
    list_filter = ['status', 'language', 'created_at']
    readonly_fields = ['created_at']
    
    def transcript_preview(self, obj):
        return obj.transcript_text[:20] + "..." if len(obj.transcript_text) > 20 else obj.transcript_text
    transcript_preview.short_description = 'Transcript Preview'