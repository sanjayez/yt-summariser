from django.contrib import admin

from yt_workflow.models import (
    CommentsResult,
    MetadataResult,
    TranscriptSegment,
    VideoTable,
    YTInsightRun,
)


@admin.register(YTInsightRun)
class YTInsightRunAdmin(admin.ModelAdmin):
    list_display = ["run_id", "query_request", "status", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["run_id", "query_request__search_id"]
    readonly_fields = ["run_id", "created_at"]
    ordering = ["-created_at"]


class BaseResultAdmin(admin.ModelAdmin):
    """Base admin configuration for all result models"""

    list_display = ["video_id", "status", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["video_id"]
    readonly_fields = ["created_at"]
    ordering = ["-created_at"]


@admin.register(VideoTable)
class VideoTableAdmin(BaseResultAdmin):
    pass


@admin.register(MetadataResult)
class MetadataResultAdmin(BaseResultAdmin):
    pass


@admin.register(TranscriptSegment)
class TranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = ["line_id", "video_id", "idx", "start", "end", "text"]
    list_filter = ["video_id"]
    search_fields = ["video_id", "text"]
    readonly_fields = ["line_id"]
    ordering = ["video_id", "idx"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related()


@admin.register(CommentsResult)
class CommentsResultAdmin(BaseResultAdmin):
    pass
