"""
Service Container for dependency injection and service management.
Provides centralized service initialization and management for the API layer.
"""
from typing import Dict, Any, Optional
from functools import lru_cache
from telemetry import get_logger


class ServiceContainer:
    """
    Simple dependency injection container for API services.
    
    Provides centralized management of service instances and dependencies,
    replacing the scattered service initialization from the original get_rag_services function.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._services: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all required services for the API layer."""
        if self._initialized:
            return
        
        self.logger.info("🚀 Initializing API service container...")
        
        try:
            # Initialize AI services from existing ai_utils package
            self._initialize_ai_services()
            
            # Initialize API-specific services
            self._initialize_api_services()
            
            self._initialized = True
            self.logger.info("✅ API service container initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize service container: {e}")
            raise
    
    def _initialize_ai_services(self) -> None:
        """Initialize AI services from ai_utils package (no OpenAI, Gemini only)."""
        from ai_utils.config import get_config
        from ai_utils.providers.gemini_llm import GeminiLLMProvider
        from ai_utils.providers.weaviate_store import WeaviateVectorStoreProvider
        from ai_utils.services.vector_service import VectorService
        from ai_utils.services.llm_service import LLMService
        
        # Get configuration
        config = get_config()
        self._services['config'] = config
        
        # Initialize vector store provider (Weaviate with native embeddings)
        vector_provider = WeaviateVectorStoreProvider(config)
        self._services['vector_provider'] = vector_provider
        
        # Initialize LLM provider (Gemini only, no OpenAI fallback)
        llm_provider = GeminiLLMProvider(config=config)
        self._services['llm_provider'] = llm_provider
        
        # Initialize services with providers
        vector_service = VectorService(provider=vector_provider)
        llm_service = LLMService(provider=llm_provider)
        
        self._services['vector'] = vector_service
        self._services['llm'] = llm_service
        
        self.logger.info("✅ AI services initialized (Gemini + Weaviate)")
    
    def _initialize_api_services(self) -> None:
        """Initialize API-specific services."""
        from .cache_service import CacheService
        from .response_service import ResponseService
        
        # Initialize API services
        cache_service = CacheService()
        response_service = ResponseService()
        
        self._services['cache'] = cache_service
        self._services['response'] = response_service
        
        self.logger.info("✅ API services initialized")
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not found or container not initialized
        """
        if not self._initialized:
            self.initialize()
        
        if service_name not in self._services:
            available = list(self._services.keys())
            raise ValueError(f"Service '{service_name}' not found. Available: {available}")
        
        return self._services[service_name]
    
    def get_ai_services(self) -> Dict[str, Any]:
        """
        Get all AI services as a dictionary (maintains compatibility with original get_rag_services).
        
        Returns:
            Dictionary with vector, llm, and config services
        """
        if not self._initialized:
            self.initialize()
        
        return {
            'vector': self._services['vector'],
            'llm': self._services['llm'],
            'config': self._services['config']
        }
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all services.
        
        Returns:
            Dictionary of all registered services
        """
        if not self._initialized:
            self.initialize()
        
        return self._services.copy()
    
    def register_service(self, name: str, service: Any) -> None:
        """
        Register a new service instance.
        
        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        self.logger.debug(f"Registered service: {name}")
    
    def is_initialized(self) -> bool:
        """Check if container is initialized."""
        return self._initialized
    
    def reset(self) -> None:
        """Reset the container (mainly for testing)."""
        self._services.clear()
        self._initialized = False
        self.logger.info("Service container reset")


# Global service container instance (singleton pattern)
_service_container: Optional[ServiceContainer] = None


@lru_cache(maxsize=1)
def get_service_container() -> ServiceContainer:
    """
    Get the global service container instance.
    
    This replaces the original get_rag_services function with a proper DI container.
    Uses LRU cache to ensure singleton behavior.
    
    Returns:
        ServiceContainer instance
    """
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


def get_rag_services() -> Dict[str, Any]:
    """
    Compatibility function that maintains the original get_rag_services interface.
    This allows existing code to work without changes while using the new service container.
    
    Returns:
        Dictionary with vector, llm, and config services
    """
    container = get_service_container()
    return container.get_ai_services()