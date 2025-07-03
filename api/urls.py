from django.urls import path
from .views import summarise_single, status_stream

urlpatterns = [
    path('url/', summarise_single, name='summarise_single'),
    path('status-stream/<uuid:request_id>/', status_stream, name='status_stream'),
    path('topic/<str:query>', summarise_single ),
]