"""
ASGI config for yt_summariser project.

Enhanced configuration for Django Channels with WebSocket and HTTP support.
Optimized for Server-Sent Events (SSE) and real-time streaming.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

# Set Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yt_summariser.settings")

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

# Import after Django setup to avoid AppRegistryNotReady errors
try:
    from django.urls import path

    # Add WebSocket URL patterns here if needed in the future
    websocket_urlpatterns = [
        # Example: path('ws/video/<str:request_id>/', VideoConsumer.as_asgi()),
    ]
except ImportError:
    websocket_urlpatterns = []

# Enhanced ASGI application with protocol routing
application = ProtocolTypeRouter(
    {
        # HTTP protocol handler (includes SSE support)
        "http": django_asgi_app,
        # WebSocket protocol handler for future real-time features
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        ),
    }
)
