from django.urls import path

from .views.gateway_views import UnifiedGatewayView
from .views.search_views import ask_video_question
from .views.status_views import video_status_stream
from .views.video_views import get_video_summary, process_single_video

urlpatterns = [
    # Unified session management endpoints
    path("process/", UnifiedGatewayView.as_view(), name="unified_process"),
    # Video processing endpoints (existing)
    path("video/process/", process_single_video, name="process_single_video"),
    path(
        "video/status/<uuid:request_id>/",
        video_status_stream,
        name="video_status_stream",
    ),
    path(
        "video/summary/<uuid:request_id>/", get_video_summary, name="get_video_summary"
    ),
    path("video/ask/<uuid:request_id>/", ask_video_question, name="ask_video_question"),
]
