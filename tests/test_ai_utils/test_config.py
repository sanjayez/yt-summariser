"""
Unit tests for configuration management.
"""

import pytest
import os
from unittest.mock import patch
from ai_utils.config import AIConfig, OpenAIConfig, PineconeConfig, get_config, update_config

class TestAIConfig:
    """Test AI configuration"""
    
    def test_openai_config_valid(self):
        """Test valid OpenAI configuration"""
        config = OpenAIConfig(
            api_key="test_key",
            model="text-embedding-3-small",
            timeout=30
        )
        assert config.api_key == "test_key"
        assert config.model == "text-embedding-3-small"
        assert config.timeout == 30
    
    def test_pinecone_config_valid(self):
        """Test valid Pinecone configuration"""
        config = PineconeConfig(
            api_key="test_key",
            environment="us-west1-gcp",
            index_name="test-index",
            dimension=1536
        )
        assert config.api_key == "test_key"
        assert config.environment == "us-west1-gcp"
        assert config.index_name == "test-index"
        assert config.dimension == 1536
    
    def test_ai_config_valid(self):
        """Test valid AI configuration"""
        config = AIConfig(
            openai=OpenAIConfig(api_key="test_openai"),
            pinecone=PineconeConfig(api_key="test_pinecone", environment="test_env"),
            batch_size=100,
            enable_async=True
        )
        assert config.batch_size == 100
        assert config.enable_async is True
        assert config.openai.api_key == "test_openai"
        assert config.pinecone.api_key == "test_pinecone"
    
    def test_ai_config_validation(self):
        """Test AI configuration validation"""
        # Should raise error for missing required fields
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            config = AIConfig(
                openai=OpenAIConfig(api_key=""),
                pinecone=PineconeConfig(api_key="test", environment="test")
            )
            config.validate()
        
        with pytest.raises(ValueError, match="PINECONE_API_KEY is required"):
            config = AIConfig(
                openai=OpenAIConfig(api_key="test"),
                pinecone=PineconeConfig(api_key="", environment="test")
            )
            config.validate()
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_openai_key",
        "PINECONE_API_KEY": "test_pinecone_key",
        "PINECONE_ENVIRONMENT": "test_env",
        "AI_BATCH_SIZE": "200",
        "AI_ENABLE_ASYNC": "false"
    })
    def test_from_env(self):
        """Test configuration from environment variables"""
        config = AIConfig.from_env()
        assert config.openai.api_key == "test_openai_key"
        assert config.pinecone.api_key == "test_pinecone_key"
        assert config.pinecone.environment == "test_env"
        assert config.batch_size == 200
        assert config.enable_async is False
    
    def test_get_config(self):
        """Test get_config function"""
        config = get_config()
        assert isinstance(config, AIConfig)
    
    def test_update_config(self):
        """Test update_config function"""
        original_batch_size = get_config().batch_size
        update_config(batch_size=500)
        assert get_config().batch_size == 500
        
        # Reset to original
        update_config(batch_size=original_batch_size) 