import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yt_summariser.settings')

app = Celery('yt_summariser')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Additional Celery configuration for task state tracking
app.conf.update(
    # Enable comprehensive task state tracking
    task_track_started=True,
    task_send_sent_event=True,
    send_events=True,
    worker_send_task_events=True,
    # Result backend reliability settings
    result_backend_always_retry=True,
    result_backend_max_retries=10,
)

app.autodiscover_tasks()