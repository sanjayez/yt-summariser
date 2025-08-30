from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.transcript_health_check, name="transcript_health_check"),
    path(
        "transcript/<uuid:request_id>/",
        views.get_transcript_with_timestamps,
        name="get_transcript_with_timestamps",
    ),
]
