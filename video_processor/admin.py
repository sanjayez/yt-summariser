from django.contrib import admin

# Register your models here.
from .models import (
    ContentAnalysis,
    TranscriptSegment,
    VideoExclusionTable,
    VideoMetadata,
    VideoTranscript,
)


@admin.register(VideoMetadata)
class VideoMetadataAdmin(admin.ModelAdmin):
    list_display = [
        "video_id",
        "title",
        "channel_name",
        "duration",
        "view_count",
        "like_count",
        "comment_count",
        "language",
        "upload_date",
        "channel_is_verified",
        "status",
        "created_at",
    ]
    list_filter = [
        "status",
        "language",
        "channel_is_verified",
        "upload_date",
        "created_at",
    ]
    search_fields = ["video_id", "title", "channel_name", "channel_id", "uploader_id"]
    readonly_fields = [
        "created_at",
        "duration_string",
        "webpage_url",
        "engagement_preview",
    ]

    fieldsets = (
        (
            "Basic Info",
            {
                "fields": (
                    "video_id",
                    "title",
                    "description",
                    "duration",
                    "duration_string",
                    "language",
                    "upload_date",
                )
            },
        ),
        (
            "Channel Info",
            {
                "fields": (
                    "channel_name",
                    "channel_id",
                    "uploader_id",
                    "channel_follower_count",
                    "channel_is_verified",
                    "channel_thumbnail",
                )
            },
        ),
        (
            "Engagement",
            {
                "fields": (
                    "view_count",
                    "like_count",
                    "comment_count",
                    "engagement",
                    "engagement_preview",
                )
            },
        ),
        ("Media", {"fields": ("thumbnail", "webpage_url")}),
        ("Categorization", {"fields": ("tags", "categories")}),
        ("System", {"fields": ("status", "created_at")}),
    )

    def engagement_preview(self, obj):
        """Display engagement segments in a readable format"""
        if not obj.engagement:
            return "No high engagement segments"

        segments = obj.engagement
        if not isinstance(segments, list) or len(segments) == 0:
            return "No high engagement segments"

        preview = []
        for segment in segments[:3]:  # Show first 3 segments
            if isinstance(segment, dict):
                start = segment.get("start_time", 0)
                end = segment.get("end_time", 0)
                value = segment.get("value", 0)
                start_min = int(start // 60)
                start_sec = int(start % 60)
                end_min = int(end // 60)
                end_sec = int(end % 60)
                preview.append(
                    f"{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d} ({value:.2f})"
                )

        result = "; ".join(preview)
        if len(segments) > 3:
            result += f"... (+{len(segments) - 3} more)"

        return result

    engagement_preview.short_description = "High Engagement Segments (>95%)"


@admin.register(VideoTranscript)
class VideoTranscriptAdmin(admin.ModelAdmin):
    list_display = [
        "video_id",
        "video_language",
        "status",
        "transcript_preview",
        "segments_count",
        "has_content_analysis",
        "created_at",
    ]
    list_filter = ["status", "created_at"]
    readonly_fields = ["created_at"]
    search_fields = ["video_id"]

    def transcript_preview(self, obj):
        return (
            obj.transcript_text[:20] + "..."
            if len(obj.transcript_text) > 20
            else obj.transcript_text
        )

    transcript_preview.short_description = "Transcript Preview"

    def segments_count(self, obj):
        return obj.segments.count()

    segments_count.short_description = "Segments"

    def video_language(self, obj):
        return obj.video_metadata.language if obj.video_metadata else "N/A"

    video_language.short_description = "Language"

    def has_content_analysis(self, obj):
        """Check if transcript has associated content analysis"""
        return hasattr(obj, "content_analysis") and obj.content_analysis is not None

    has_content_analysis.boolean = True
    has_content_analysis.short_description = "Content Analysis"


@admin.register(TranscriptSegment)
class TranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = [
        "segment_id",
        "transcript",
        "sequence_number",
        "start_time",
        "end_time",
        "is_embedded",
        "text_preview",
    ]
    list_filter = ["is_embedded", "transcript__video_id"]
    search_fields = ["segment_id", "text", "transcript__video_id"]
    readonly_fields = ["created_at"]
    ordering = ["transcript", "sequence_number"]

    def text_preview(self, obj):
        return obj.text[:50] + "..." if len(obj.text) > 50 else obj.text

    text_preview.short_description = "Text Preview"


@admin.register(VideoExclusionTable)
class VideoExclusionTableAdmin(admin.ModelAdmin):
    list_display = [
        "video_id",
        "exclusion_reason_display",
        "video_url_short",
        "detected_at",
    ]
    list_filter = ["exclusion_reason", "detected_at"]
    search_fields = ["video_id", "video_url"]
    readonly_fields = ["detected_at", "video_id_clickable"]
    ordering = ["-detected_at"]

    fieldsets = (
        (
            "Video Information",
            {"fields": ("video_id", "video_id_clickable", "video_url")},
        ),
        ("Exclusion Details", {"fields": ("exclusion_reason", "detected_at")}),
    )

    def video_url_short(self, obj):
        """Display shortened video URL for better list view"""
        return f"{obj.video_url[:50]}..." if len(obj.video_url) > 50 else obj.video_url

    video_url_short.short_description = "Video URL"

    def exclusion_reason_display(self, obj):
        """Display exclusion reason with nice formatting"""
        return obj.get_exclusion_reason_display()

    exclusion_reason_display.short_description = "Exclusion Reason"

    def video_id_clickable(self, obj):
        """Make video ID clickable to open YouTube in new tab"""
        if obj.video_id:
            youtube_url = f"https://www.youtube.com/watch?v={obj.video_id}"
            return f'<a href="{youtube_url}" target="_blank">{obj.video_id}</a>'
        return obj.video_id

    video_id_clickable.short_description = "Video ID (Clickable)"
    video_id_clickable.allow_tags = True

    def get_queryset(self, request):
        """Optimize queries for video exclusions"""
        return super().get_queryset(request)

    # Add custom admin actions
    actions = ["export_exclusion_analytics"]

    def export_exclusion_analytics(self, request, queryset):
        """Export exclusion analytics data"""
        import csv

        from django.http import HttpResponse

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="video_exclusions.csv"'

        writer = csv.writer(response)
        writer.writerow(["Video ID", "Video URL", "Exclusion Reason", "Detected At"])

        for exclusion in queryset:
            writer.writerow(
                [
                    exclusion.video_id,
                    exclusion.video_url,
                    exclusion.get_exclusion_reason_display(),
                    exclusion.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        self.message_user(request, f"Exported {queryset.count()} exclusion records.")
        return response

    export_exclusion_analytics.short_description = "Export selected exclusions to CSV"


@admin.register(ContentAnalysis)
class ContentAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        "video_id",
        "preliminary_status_display",
        "timestamped_status_display",
        "content_rating_display",
        "segments_summary",
        "speaker_tones_display",
        "created_at",
    ]
    list_filter = [
        "preliminary_analysis_status",
        "timestamped_analysis_status",
        "created_at",
    ]
    search_fields = ["video_transcript__video_id"]
    readonly_fields = [
        "created_at",
        "updated_at",
        "preliminary_completed_at",
        "final_completed_at",
        "analysis_summary",
        "video_link",
    ]

    fieldsets = (
        ("Video Info", {"fields": ("video_transcript", "video_link")}),
        (
            "Phase 1: Preliminary Analysis",
            {
                "fields": (
                    "preliminary_analysis_status",
                    "preliminary_completed_at",
                    "speaker_tones",
                    "raw_ad_segments",
                    "raw_filler_segments",
                )
            },
        ),
        (
            "Phase 2: Timestamped Analysis",
            {
                "fields": (
                    "timestamped_analysis_status",
                    "final_completed_at",
                    "content_rating",
                    "ad_duration_ratio",
                    "filler_duration_ratio",
                    "ad_segments",
                    "filler_segments",
                    "content_segments",
                )
            },
        ),
        ("Analysis Summary", {"fields": ("analysis_summary",)}),
        ("System", {"fields": ("created_at", "updated_at")}),
    )

    def video_id(self, obj):
        """Display video ID from related transcript"""
        return obj.video_transcript.video_id if obj.video_transcript else "N/A"

    video_id.short_description = "Video ID"
    video_id.admin_order_field = "video_transcript__video_id"

    def preliminary_status_display(self, obj):
        """Display preliminary status with color coding"""
        status = obj.preliminary_analysis_status
        colors = {
            "pending": "gray",
            "processing": "orange",
            "completed": "green",
            "failed": "red",
        }
        color = colors.get(status, "black")
        return (
            f'<span style="color: {color}; font-weight: bold;">{status.upper()}</span>'
        )

    preliminary_status_display.short_description = "Phase 1 Status"
    preliminary_status_display.allow_tags = True

    def timestamped_status_display(self, obj):
        """Display timestamped status with color coding"""
        status = obj.timestamped_analysis_status
        colors = {
            "pending": "gray",
            "processing": "orange",
            "completed": "green",
            "failed": "red",
        }
        color = colors.get(status, "black")
        return (
            f'<span style="color: {color}; font-weight: bold;">{status.upper()}</span>'
        )

    timestamped_status_display.short_description = "Phase 2 Status"
    timestamped_status_display.allow_tags = True

    def content_rating_display(self, obj):
        """Display content rating with color coding"""
        if obj.content_rating is None:
            return '<span style="color: gray;">N/A</span>'

        rating = obj.content_rating
        if rating >= 0.8:
            color = "green"
        elif rating >= 0.6:
            color = "orange"
        else:
            color = "red"

        return f'<span style="color: {color}; font-weight: bold;">{rating:.3f}</span>'

    content_rating_display.short_description = "Content Rating"
    content_rating_display.allow_tags = True

    def segments_summary(self, obj):
        """Display summary of segments"""
        if not obj.is_final_complete:
            return '<span style="color: gray;">Analysis incomplete</span>'

        ad_count = len(obj.ad_segments) if obj.ad_segments else 0
        filler_count = len(obj.filler_segments) if obj.filler_segments else 0
        content_count = len(obj.content_segments) if obj.content_segments else 0

        return f"Ads: {ad_count} | Filler: {filler_count} | Content: {content_count}"

    segments_summary.short_description = "Segments"
    segments_summary.allow_tags = True

    def speaker_tones_display(self, obj):
        """Display speaker tones as readable list"""
        if not obj.speaker_tones:
            return "None detected"
        return ", ".join(obj.speaker_tones[:3]) + (
            "..." if len(obj.speaker_tones) > 3 else ""
        )

    speaker_tones_display.short_description = "Speaker Tones"

    def analysis_summary(self, obj):
        """Display comprehensive analysis summary"""
        summary = obj.get_analysis_summary()

        if summary["status"] == "incomplete":
            return f"""Analysis Status: {summary["status"].upper()}
Preliminary Complete: {summary["preliminary_complete"]}
Final Complete: {summary["final_complete"]}"""

        return f"""Analysis Status: {summary["status"].upper()}
Content Rating: {summary["content_rating"]:.3f}
Ad Ratio: {summary["ad_ratio"]:.3f}
Filler Ratio: {summary["filler_ratio"]:.3f}
Speaker Tones: {", ".join(summary["speaker_tones"])}
Total Ad Segments: {summary["total_ad_segments"]}
Total Filler Segments: {summary["total_filler_segments"]}"""

    analysis_summary.short_description = "Analysis Summary"

    def video_link(self, obj):
        """Display clickable YouTube link"""
        if obj.video_transcript and obj.video_transcript.video_id:
            video_id = obj.video_transcript.video_id
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            return f'<a href="{youtube_url}" target="_blank">Watch on YouTube</a>'
        return "N/A"

    video_link.short_description = "YouTube Link"
    video_link.allow_tags = True

    def get_queryset(self, request):
        """Optimize queries for content analysis"""
        return super().get_queryset(request).select_related("video_transcript")

    # Custom admin actions
    actions = ["export_content_analysis_csv", "retry_failed_analysis"]

    def export_content_analysis_csv(self, request, queryset):
        """Export content analysis data to CSV"""
        import csv

        from django.http import HttpResponse

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="content_analysis.csv"'

        writer = csv.writer(response)
        writer.writerow(
            [
                "Video ID",
                "Preliminary Status",
                "Timestamped Status",
                "Content Rating",
                "Ad Ratio",
                "Filler Ratio",
                "Speaker Tones",
                "Ad Segments Count",
                "Filler Segments Count",
                "Created At",
            ]
        )

        for analysis in queryset:
            writer.writerow(
                [
                    analysis.video_transcript.video_id
                    if analysis.video_transcript
                    else "N/A",
                    analysis.preliminary_analysis_status,
                    analysis.timestamped_analysis_status,
                    analysis.content_rating or 0,
                    analysis.ad_duration_ratio or 0,
                    analysis.filler_duration_ratio or 0,
                    ", ".join(analysis.speaker_tones) if analysis.speaker_tones else "",
                    len(analysis.ad_segments) if analysis.ad_segments else 0,
                    len(analysis.filler_segments) if analysis.filler_segments else 0,
                    analysis.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        self.message_user(
            request, f"Exported {queryset.count()} content analysis records."
        )
        return response

    export_content_analysis_csv.short_description = "Export selected analyses to CSV"

    def retry_failed_analysis(self, request, queryset):
        """Reset failed analyses to pending for retry"""
        failed_analyses = queryset.filter(preliminary_analysis_status="failed").update(
            preliminary_analysis_status="pending"
        )

        failed_timestamped = queryset.filter(
            timestamped_analysis_status="failed"
        ).update(timestamped_analysis_status="pending")

        total_reset = failed_analyses + failed_timestamped
        self.message_user(
            request, f"Reset {total_reset} failed analyses to pending status."
        )

    retry_failed_analysis.short_description = "Reset failed analyses to pending"
