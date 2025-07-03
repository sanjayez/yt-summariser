from django.contrib import admin
from .models import URLRequestTable

# Register your models here.

class URLRequestTableAdmin(admin.ModelAdmin):
    list_display = ['request_id_short', 'url', 'ip_address', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['url', 'ip_address']
    readonly_fields = ['request_id', 'created_at']
    
    def request_id_short(self, obj):
        return str(obj.request_id)[:8]
    request_id_short.short_description = 'Request ID'

admin.site.register(URLRequestTable, URLRequestTableAdmin)
