from django.contrib import admin
from .models import SearchSession, SearchRequest


@admin.register(SearchSession)
class SearchSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id_short', 'user_ip', 'status', 'requests_count', 'created_at']
    list_filter = ['status', 'created_at', 'user_ip']
    search_fields = ['session_id', 'user_ip']
    readonly_fields = ['session_id', 'created_at']
    ordering = ['-created_at']
    
    def session_id_short(self, obj):
        return str(obj.session_id)[:8]
    session_id_short.short_description = 'Session ID'
    
    def requests_count(self, obj):
        return obj.search_requests.count()
    requests_count.short_description = 'Requests'


@admin.register(SearchRequest)
class SearchRequestAdmin(admin.ModelAdmin):
    list_display = ['search_id_short', 'session_short', 'original_query_preview', 'processed_query_preview', 'total_videos', 'status', 'created_at']
    list_filter = ['status', 'created_at', 'search_session__user_ip', 'total_videos']
    search_fields = ['search_id', 'original_query', 'processed_query', 'search_session__session_id']
    readonly_fields = ['search_id', 'created_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('search_id', 'search_session', 'status', 'created_at')
        }),
        ('Search Queries', {
            'fields': ('original_query', 'processed_query')
        }),
        ('Results', {
            'fields': ('video_urls', 'total_videos')
        }),
        ('Error Info', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
    )
    
    def search_id_short(self, obj):
        return str(obj.search_id)[:8]
    search_id_short.short_description = 'Search ID'
    
    def session_short(self, obj):
        return str(obj.search_session.session_id)[:8] if obj.search_session else 'N/A'
    session_short.short_description = 'Session'
    
    def original_query_preview(self, obj):
        return obj.original_query[:50] + "..." if len(obj.original_query) > 50 else obj.original_query
    original_query_preview.short_description = 'Original Query'
    
    def processed_query_preview(self, obj):
        if not obj.processed_query:
            return 'N/A'
        return obj.processed_query[:50] + "..." if len(obj.processed_query) > 50 else obj.processed_query
    processed_query_preview.short_description = 'Processed Query'
