from django.contrib import admin

from yt_workflow.models import (
    CommentsResult,
    MetadataResult,
    TranscriptResult,
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


@admin.register(TranscriptResult)
class TranscriptResultAdmin(BaseResultAdmin):
    pass


@admin.register(CommentsResult)
class CommentsResultAdmin(BaseResultAdmin):
    pass
