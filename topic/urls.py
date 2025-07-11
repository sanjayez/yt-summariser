from django.urls import path
from . import views

urlpatterns = [
    # Unified search and process endpoint
    path('search/', views.IntegratedSearchProcessAPIView.as_view(), name='topic_search_process'),
    
    # Status tracking endpoint
    path('search/status/<uuid:search_request_id>/', views.SearchStatusAPIView.as_view(), name='search_status'),
]