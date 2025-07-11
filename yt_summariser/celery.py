import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yt_summariser.settings')

app = Celery('yt_summariser')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Force django-celery-results configuration with extended information
app.conf.update(
    result_backend='django-db',
    cache_backend='django-cache',
    result_expires=60 * 60 * 24 * 7,  # 7 days
    result_extended=True,  # Store task name, args, kwargs, worker info
    result_backend_always_retry=True,
    result_backend_max_retries=10,
)

app.autodiscover_tasks()