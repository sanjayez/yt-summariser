"""
Query Processor App Configuration
"""

from django.apps import AppConfig


class QueryProcessorConfig(AppConfig):
    """Configuration for the query_processor Django app"""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'query_processor'
    verbose_name = 'Query Processor'
    
    def ready(self):
        """
        Perform initialization when the app is ready.
        This is called once Django has loaded all apps.
        """
        # Import any signal handlers or perform app initialization here
        pass