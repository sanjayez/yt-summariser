from django.urls import path
from .views.video_views import process_single_video, get_video_summary
from .views.status_views import video_status_stream
from .views.search_views import ask_video_question

urlpatterns = [
    # Video processing endpoints
    path('video/process/', process_single_video, name='process_single_video'),
    path('video/status/<uuid:request_id>/', video_status_stream, name='video_status_stream'),
    path('video/summary/<uuid:request_id>/', get_video_summary, name='get_video_summary'),
    path('video/ask/<uuid:request_id>/', ask_video_question, name='ask_video_question'),
]