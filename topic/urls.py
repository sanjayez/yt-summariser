from django.urls import path
from . import views

urlpatterns = [
    # Unified search and process endpoint
    path('search/', views.IntegratedSearchProcessAPIView.as_view(), name='topic_search_process'),
    
    # SSE endpoints
    path('test/sse/', views.simple_sse_test, name='simple_sse_test'),
    path('search/status/<uuid:search_id>/stream/', views.search_status_stream, name='search_status_stream'),
]