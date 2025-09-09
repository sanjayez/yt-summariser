from django.urls import path

from . import views

urlpatterns = [
    path("test/<str:video_id>/", views.test_metadata_fetch, name="test_metadata_fetch"),
]
