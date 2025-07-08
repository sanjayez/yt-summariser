from django.urls import path
from . import views

urlpatterns = [
    path('search/', views.TopicSearchAPIView.as_view(), name='search_topic'),
]