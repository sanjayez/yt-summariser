from django.contrib import admin
from django.db.models import CharField
from django.db.models.functions import Cast
from .models import QueryRequest


@admin.register(QueryRequest)
class QueryRequestAdmin(admin.ModelAdmin):
    list_display = ("search_id", "request_type", "status", "total_videos", "created_at")
    # Keep text-only to avoid DB operator errors on UUIDField
    search_fields = ("original_content",)
    list_filter = ("request_type", "status", "intent_type", "created_at")
    date_hierarchy = "created_at"
    ordering = ("-created_at",)
    readonly_fields = ("search_id", "created_at")

    # to avoid DB operator errors on UUIDField
    def get_search_results(self, request, queryset, search_term):
        # Start with default search (original_content)
        queryset, use_distinct = super().get_search_results(request, queryset, search_term)
        if search_term:
            # OR-search on casted UUID text
            by_id = self.model.objects.annotate(
                search_id_text=Cast("search_id", CharField())
            ).filter(search_id_text__icontains=search_term)
            queryset = queryset.union(by_id)
        return queryset, use_distinct
