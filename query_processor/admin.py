from django.contrib import admin
from .models import QueryRequest


@admin.register(QueryRequest)
class QueryRequestAdmin(admin.ModelAdmin):
    list_display = ("search_id", "request_type", "status", "total_videos", "created_at")
    search_fields = ("search_id", "original_content")
    list_filter = ("request_type", "status", "intent_type", "created_at")
    date_hierarchy = "created_at"
    ordering = ("-created_at",)
    readonly_fields = ("search_id", "created_at")
