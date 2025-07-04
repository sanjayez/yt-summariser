from django.contrib import admin

# Register your models here.
from .models import VideoMetadata, VideoTranscript, TranscriptSegment

@admin.register(VideoMetadata)
class VideoMetadataAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'title', 'channel_name', 'duration', 'view_count', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['video_id', 'title', 'channel_name']
    readonly_fields = ['created_at']

@admin.register(VideoTranscript)
class VideoTranscriptAdmin(admin.ModelAdmin):
    list_display = ['video_metadata', 'language', 'status', 'transcript_preview', 'segments_count', 'created_at']
    list_filter = ['status', 'language', 'created_at']
    readonly_fields = ['created_at']
    
    def transcript_preview(self, obj):
        return obj.transcript_text[:20] + "..." if len(obj.transcript_text) > 20 else obj.transcript_text
    transcript_preview.short_description = 'Transcript Preview'
    
    def segments_count(self, obj):
        return obj.segments.count()
    segments_count.short_description = 'Segments'

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