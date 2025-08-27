from django.contrib import admin
from .models import URLRequestTable, UnifiedSession
from django_celery_results.models import TaskResult

# Register your models here.

class UnifiedSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id_short', 'user_ip', 'total_requests_display', 'video_requests', 'playlist_requests', 'topic_requests', 'created_at', 'last_request_at']
    list_filter = ['created_at', 'last_request_at']
    search_fields = ['user_ip', '^session_id']
    readonly_fields = ['session_id', 'created_at', 'last_request_at', 'total_requests_display',
                       'video_requests', 'playlist_requests', 'topic_requests']
    ordering = ['-last_request_at']
    
    fieldsets = (
        ('Session Info', {
            'fields': ('session_id', 'user_ip', 'user_account')
        }),
        ('Request Counters', {
            'fields': ('video_requests', 'playlist_requests', 'topic_requests', 'total_requests_display')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'last_request_at')
        }),
    )
    
    def session_id_short(self, obj):
        return str(obj.session_id)[:8]
    session_id_short.short_description = 'Session ID'
    
    def total_requests_display(self, obj):
        return f"{obj.total_requests}/3"
    total_requests_display.short_description = 'Total Requests'

class URLRequestTableAdmin(admin.ModelAdmin):
    list_display = ['request_id_short', 'url', 'ip_address', 'status', 'failure_reason', 'created_at']
    list_filter = ['status', 'failure_reason', 'created_at']
    search_fields = ['url', 'ip_address']
    readonly_fields = ['request_id', 'created_at']
    
    def request_id_short(self, obj):
        return str(obj.request_id)[:8]
    request_id_short.short_description = 'Request ID'

# Customize existing TaskResult admin
class CustomTaskResultAdmin(admin.ModelAdmin):
    list_display = ['task_id_short', 'task_name', 'status', 'date_created', 'date_done', 'worker']
    list_filter = ['status', 'task_name', 'date_created', 'worker']
    search_fields = ['task_id', 'task_name', 'status']
    readonly_fields = ['task_id', 'task_name', 'status', 'result', 'traceback', 'date_created', 'date_done', 'worker']
    ordering = ['-date_created']
    
    def task_id_short(self, obj):
        return str(obj.task_id)[:8] + '...'
    task_id_short.short_description = 'Task ID'

# Re-register with custom admin
admin.site.unregister(TaskResult)
admin.site.register(TaskResult, CustomTaskResultAdmin)
admin.site.register(URLRequestTable, URLRequestTableAdmin)
admin.site.register(UnifiedSession, UnifiedSessionAdmin)
