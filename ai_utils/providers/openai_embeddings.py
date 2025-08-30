"""
OpenAI Embedding Provider Implementation
Handles text embedding using OpenAI's API with async support, batching, and retry logic
"""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

from ..config import AIConfig
from ..interfaces.embeddings import EmbeddingProvider
from ..models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with async support and batching"""

    def __init__(self, config: AIConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
            timeout=config.openai.timeout,
            max_retries=config.openai.max_retries,
        )
        self.batch_size = config.batch_size
        self.max_concurrent = config.max_concurrent_requests

    async def embed_text(
        self, text_or_request: str | EmbeddingRequest
    ) -> EmbeddingResponse:
        """Embed a single text string or request"""
        try:
            # Handle both string and EmbeddingRequest inputs
            if isinstance(text_or_request, str):
                text = text_or_request
                model = self.config.openai.embedding_model
                user = self.config.openai.user
            else:
                text = text_or_request.text
                model = text_or_request.model or self.config.openai.embedding_model
                user = text_or_request.user or self.config.openai.user

            # Prepare API call
            response = await self.client.embeddings.create(
                model=model, input=text, user=user
            )

            # Extract embedding data
            embedding_data = response.data[0]

            return EmbeddingResponse(
                embedding=embedding_data.embedding,
                model=response.model,
                usage=response.usage.model_dump(),
            )

        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise

    async def embed_batch(
        self, batch_request: BatchEmbeddingRequest
    ) -> BatchEmbeddingResponse:
        """Embed multiple texts in a batch"""
        try:
            # Handle the batch request
            texts = batch_request.texts
            model = batch_request.model or self.config.openai.embedding_model
            user = batch_request.user or self.config.openai.user

            # If batch is small, process directly
            if len(texts) <= self.batch_size:
                return await self._embed_batch_direct(texts, model, user)

            # For larger batches, split into chunks
            embeddings = []
            total_usage = {"prompt_tokens": 0, "total_tokens": 0}

            # Process in chunks
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                chunk_response = await self._embed_batch_direct(chunk, model, user)
                embeddings.extend(chunk_response.embeddings)

                # Accumulate usage
                total_usage["prompt_tokens"] += chunk_response.usage.get(
                    "prompt_tokens", 0
                )
                total_usage["total_tokens"] += chunk_response.usage.get(
                    "total_tokens", 0
                )

            return BatchEmbeddingResponse(
                embeddings=embeddings, model=model, usage=total_usage
            )

        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            raise

    async def _embed_batch_direct(
        self, texts: list[str], model: str, user: str | None
    ) -> BatchEmbeddingResponse:
        """Direct batch embedding for smaller batches"""
        try:
            response = await self.client.embeddings.create(
                model=model, input=texts, user=user
            )

            # Extract embeddings in order
            embeddings = [data.embedding for data in response.data]

            return BatchEmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                usage=response.usage.model_dump(),
            )

        except Exception as e:
            logger.error(f"Error in direct batch embedding: {str(e)}")
            raise

    async def embed_texts_concurrent(
        self, texts: list[str], model: str | None = None, user: str | None = None
    ) -> list[list[float]]:
        """Embed multiple texts concurrently with rate limiting"""
        try:
            # Limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def embed_single(text: str) -> list[float]:
                async with semaphore:
                    request = EmbeddingRequest(
                        text=text,
                        model=model or self.config.openai.embedding_model,
                        user=user,
                    )
                    response = await self.embed_text(request)
                    return response.embedding

            # Execute concurrent embeddings
            tasks = [embed_single(text) for text in texts]
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            valid_embeddings = []
            for i, embedding in enumerate(embeddings):
                if isinstance(embedding, Exception):
                    logger.error(f"Error embedding text {i}: {str(embedding)}")
                    # Use zero vector as fallback
                    valid_embeddings.append([0.0] * 1536)  # Default dimension
                else:
                    valid_embeddings.append(embedding)

            return valid_embeddings

        except Exception as e:
            logger.error(f"Error in concurrent embedding: {str(e)}")
            raise

    def get_supported_models(self) -> list[str]:
        """Get list of supported embedding models"""
        return [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

    def get_model_dimensions(self, model: str) -> int:
        """Get the dimension of a specific model"""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if model not in model_dimensions:
            raise ValueError(f"Unsupported model: {model}")

        return model_dimensions[model]

    async def health_check(self) -> bool:
        """Check if the embedding provider is healthy"""
        try:
            # Test with a simple embedding
            test_response = await self.embed_text("test")
            return len(test_response.embedding) > 0

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    async def get_embedding_dimension(self, model: str | None = None) -> int:
        """Get embedding dimension for a specific model"""
        model = model or self.config.openai.embedding_model

        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return model_dimensions.get(model, 1536)  # Default to 1536 if unknown

    async def estimate_cost(
        self, text_count: int, model: str | None = None
    ) -> dict[str, Any]:
        """Estimate cost for embedding operations"""
        model = model or self.config.openai.embedding_model

        # Rough cost estimates (tokens per 1000 characters, cost per 1M tokens)
        model_costs = {
            "text-embedding-3-small": {
                "tokens_per_1k_chars": 300,
                "cost_per_1m_tokens": 0.02,
            },
            "text-embedding-3-large": {
                "tokens_per_1k_chars": 300,
                "cost_per_1m_tokens": 0.13,
            },
            "text-embedding-ada-002": {
                "tokens_per_1k_chars": 400,
                "cost_per_1m_tokens": 0.10,
            },
        }

        model_info = model_costs.get(model, model_costs["text-embedding-3-small"])

        # Rough estimate assuming 1000 characters per text
        estimated_tokens = text_count * model_info["tokens_per_1k_chars"]
        estimated_cost = (estimated_tokens / 1_000_000) * model_info[
            "cost_per_1m_tokens"
        ]

        return {
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": model,
        }
