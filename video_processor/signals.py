from django.db.models.signals import post_save
from django.dispatch import receiver
from api.models import URLRequestTable
from .tasks import process_youtube_video

@receiver(post_save, sender=URLRequestTable)
def trigger_video_processing(sender, instance, created, **kwargs):
    if created and instance.status == 'processing':  # Only trigger for new records with processing status
        process_youtube_video.delay(instance.id)