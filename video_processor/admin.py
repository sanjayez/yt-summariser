from django.contrib import admin

# Register your models here.
from .models import VideoMetadata, VideoTranscript, TranscriptSegment, VideoExclusionTable

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
    list_display = ['video_id', 'video_language', 'status', 'transcript_preview', 'segments_count', 'created_at']
    list_filter = ['status', 'created_at']
    readonly_fields = ['created_at']
    search_fields = ['video_id']
    
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
    list_filter = ['is_embedded', 'transcript__video_id']
    search_fields = ['segment_id', 'text', 'transcript__video_id']
    readonly_fields = ['created_at']
    ordering = ['transcript', 'sequence_number']
    
    def text_preview(self, obj):
        return obj.text[:50] + "..." if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Text Preview'


@admin.register(VideoExclusionTable)
class VideoExclusionTableAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'exclusion_reason_display', 'video_url_short', 'detected_at']
    list_filter = ['exclusion_reason', 'detected_at']
    search_fields = ['video_id', 'video_url']
    readonly_fields = ['detected_at', 'video_id_clickable']
    ordering = ['-detected_at']
    
    fieldsets = (
        ('Video Information', {
            'fields': ('video_id', 'video_id_clickable', 'video_url')
        }),
        ('Exclusion Details', {
            'fields': ('exclusion_reason', 'detected_at')
        }),
    )
    
    def video_url_short(self, obj):
        """Display shortened video URL for better list view"""
        return f"{obj.video_url[:50]}..." if len(obj.video_url) > 50 else obj.video_url
    video_url_short.short_description = 'Video URL'
    
    def exclusion_reason_display(self, obj):
        """Display exclusion reason with nice formatting"""
        return obj.get_exclusion_reason_display()
    exclusion_reason_display.short_description = 'Exclusion Reason'
    
    def video_id_clickable(self, obj):
        """Make video ID clickable to open YouTube in new tab"""
        if obj.video_id:
            youtube_url = f"https://www.youtube.com/watch?v={obj.video_id}"
            return f'<a href="{youtube_url}" target="_blank">{obj.video_id}</a>'
        return obj.video_id
    video_id_clickable.short_description = 'Video ID (Clickable)'
    video_id_clickable.allow_tags = True
    
    def get_queryset(self, request):
        """Optimize queries for video exclusions"""
        return super().get_queryset(request)
    
    # Add custom admin actions
    actions = ['export_exclusion_analytics']
    
    def export_exclusion_analytics(self, request, queryset):
        """Export exclusion analytics data"""
        from django.http import HttpResponse
        import csv
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="video_exclusions.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Video ID', 'Video URL', 'Exclusion Reason', 'Detected At'])
        
        for exclusion in queryset:
            writer.writerow([
                exclusion.video_id,
                exclusion.video_url,
                exclusion.get_exclusion_reason_display(),
                exclusion.detected_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        self.message_user(request, f"Exported {queryset.count()} exclusion records.")
        return response
    export_exclusion_analytics.short_description = "Export selected exclusions to CSV"