from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        # Non-blocking Gemini warmup on Django app start (best-effort)
        try:
            import threading
            import asyncio
            from ai_utils.services.registry import warmup_gemini_llm, warmup_vector_store
            from .services.service_container import get_service_container

            def _warm():
                try:
                    # Initialize the service container to construct singletons early
                    try:
                        container = get_service_container()
                        container.initialize()
                    except Exception:
                        pass
                    # Warm up LLM and vector store
                    asyncio.run(warmup_gemini_llm())
                    asyncio.run(warmup_vector_store())
                except Exception:
                    pass

            threading.Thread(target=_warm, daemon=True).start()
        except Exception:
            # App startup must not be impacted
            pass
