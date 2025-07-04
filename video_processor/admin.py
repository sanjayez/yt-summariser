from django.contrib import admin

# Register your models here.
from .models import VideoMetadata, VideoTranscript, TranscriptSegment

@admin.register(VideoMetadata)
class VideoMetadataAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'title', 'channel_name', 'duration', 'view_count', 'like_count', 'language', 'upload_date', 'channel_is_verified', 'status', 'created_at']
    list_filter = ['status', 'language', 'channel_is_verified', 'upload_date', 'created_at']
    search_fields = ['video_id', 'title', 'channel_name', 'channel_id', 'uploader_id']
    readonly_fields = ['created_at', 'duration_string', 'webpage_url']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('video_id', 'title', 'description', 'duration', 'duration_string', 'language', 'upload_date')
        }),
        ('Channel Info', {
            'fields': ('channel_name', 'channel_id', 'uploader_id', 'channel_follower_count', 'channel_is_verified')
        }),
        ('Engagement', {
            'fields': ('view_count', 'like_count')
        }),
        ('Media', {
            'fields': ('thumbnail', 'webpage_url')
        }),
        ('Categorization', {
            'fields': ('tags', 'categories')
        }),
        ('System', {
            'fields': ('status', 'created_at')
        }),
    )

@admin.register(VideoTranscript)
class VideoTranscriptAdmin(admin.ModelAdmin):
    list_display = ['video_metadata', 'video_language', 'status', 'transcript_preview', 'segments_count', 'created_at']
    list_filter = ['status', 'video_metadata__language', 'created_at']
    readonly_fields = ['created_at']
    
    def transcript_preview(self, obj):
        return obj.transcript_text[:20] + "..." if len(obj.transcript_text) > 20 else obj.transcript_text
    transcript_preview.short_description = 'Transcript Preview'
    
    def segments_count(self, obj):
        return obj.segments.count()
    segments_count.short_description = 'Segments'
    
    def video_language(self, obj):
        return obj.video_metadata.language if obj.video_metadata else 'N/A'
    video_language.short_description = 'Language'

@admin.register(TranscriptSegment)
class TranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = ['segment_id', 'transcript', 'sequence_number', 'start_time', 'end_time', 'is_embedded', 'text_preview']
    list_filter = ['is_embedded', 'transcript__video_metadata__video_id']
    search_fields = ['segment_id', 'text', 'transcript__video_metadata__video_id']
    readonly_fields = ['created_at']
    ordering = ['transcript', 'sequence_number']
    
    def text_preview(self, obj):
        return obj.text[:50] + "..." if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Text Preview'