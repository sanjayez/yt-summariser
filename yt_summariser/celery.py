import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yt_summariser.settings')

app = Celery('yt_summariser')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()