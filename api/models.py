from django.db import models
import uuid

# Create your models here.

class URLRequestTable(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    
    request_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    url = models.URLField(max_length=500)
    ip_address = models.GenericIPAddressField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Request {str(self.request_id)[:8]}"
