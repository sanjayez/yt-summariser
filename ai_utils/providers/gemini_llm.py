"""
Gemini LLM Provider Implementation using LlamaIndex
Provides a wrapper around LlamaIndex's GoogleGenerativeAI for seamless integration
"""

import contextlib
import logging
import os
import time
from typing import Any

from temp_file_logger import append_line

from ..config import AIConfig
from ..interfaces.llm import LLMProvider
from ..models import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    ChatUsage,
    RAGQuery,
    RAGResponse,
    TextGenerationRequest,
    TextGenerationResponse,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)

# Import LlamaIndex Gemini integration
try:
    from llama_index.core.base.llms.types import MessageRole
    from llama_index.core.llms import ChatMessage as LIMessage
    from llama_index.llms.google_genai import GoogleGenAI

    GEMINI_AVAILABLE = True
except ImportError as e:
    logger.error(f"LlamaIndex Google GenAI not available: {e}")
    logger.error("Install with: pip install llama-index-llms-google-genai")
    GEMINI_AVAILABLE = False
    GoogleGenAI = None  # type: ignore
    LIMessage = None  # type: ignore
    MessageRole = None  # type: ignore


class GeminiLLMProvider(LLMProvider):
    """Gemini LLM provider using LlamaIndex GoogleGenerativeAI integration"""

    def __init__(self, config: AIConfig):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "LlamaIndex Google GenAI integration not available. "
                "Install with: pip install llama-index-llms-google-genai"
            )

        self.config = config
        # Cache of subclients keyed by (model, temperature, max_tokens)
        self._client_cache: dict[tuple, Any] = {}
        # Preload default client
        self._client_cache[
            (config.gemini.model, config.gemini.temperature, config.gemini.max_tokens)
        ] = GoogleGenAI(
            model=config.gemini.model,
            api_key=config.gemini.api_key,
            temperature=config.gemini.temperature,
            max_tokens=config.gemini.max_tokens or 8000,
            timeout=config.gemini.timeout,
        )

        logger.info(
            f"Initialized Gemini LLM Provider with model: {config.gemini.model}"
        )

    def get_llm_client(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GoogleGenAI:
        """Get or create a GoogleGenAI client with specified parameters"""
        selected_model = model or self.config.gemini.model
        selected_temp = (
            temperature if temperature is not None else self.config.gemini.temperature
        )
        selected_max_tokens = (
            max_tokens if max_tokens is not None else self.config.gemini.max_tokens
        )

        key = (selected_model, selected_temp, selected_max_tokens)
        client = self._client_cache.get(key)

        if client is None:
            client = GoogleGenAI(
                model=selected_model,
                api_key=self.config.gemini.api_key,
                temperature=selected_temp,
                max_tokens=selected_max_tokens or 8000,
                timeout=self.config.gemini.timeout,
            )
            self._client_cache[key] = client

        return client

    async def generate_rag_response(
        self, query: RAGQuery, context_documents: list[VectorSearchResult]
    ) -> RAGResponse:
        """
        Generate a RAG response using context documents.
        Note: Not implemented since we are using OpenAI for RAG
        """
        logger.warning(
            "generate_rag_response() not fully implemented - user requested minimal implementation"
        )
        raise NotImplementedError(
            "generate_rag_response() not implemented for GeminiLLMProvider. "
            "Use generate_text() instead for Gemini-based text generation."
        )

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        """Generate text from a prompt using LlamaIndex Gemini integration"""
        try:
            # Configure generation parameters
            selected_model = model or self.config.gemini.model
            key = (
                selected_model,
                temperature
                if temperature is not None
                else self.config.gemini.temperature,
                max_tokens if max_tokens is not None else self.config.gemini.max_tokens,
            )
            gemini_llm = self._client_cache.get(key)
            if gemini_llm is None:
                gemini_llm = GoogleGenAI(
                    model=key[0],
                    api_key=self.config.gemini.api_key,
                    temperature=key[1],
                    max_tokens=key[2] or self.config.gemini.max_tokens,
                    timeout=self.config.gemini.timeout,
                )
                self._client_cache[key] = gemini_llm

            # Handle system prompt by prepending to user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Use LlamaIndex's async complete method
            response = await gemini_llm.acomplete(full_prompt)

            return str(response)

        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Complete chat conversation using LlamaIndex Gemini chat interface"""
        try:
            # Convert our ChatMessage format to LlamaIndex format
            li_messages = []
            for msg in request.messages:
                # Map our ChatRole to LlamaIndex MessageRole
                if msg.role == ChatRole.USER:
                    role = MessageRole.USER
                elif msg.role == ChatRole.ASSISTANT:
                    role = MessageRole.ASSISTANT
                elif msg.role == ChatRole.SYSTEM:
                    role = MessageRole.SYSTEM
                else:
                    role = MessageRole.USER  # Default fallback

                li_messages.append(LIMessage(role=role, content=msg.content))

            # Configure Gemini with request parameters if different from defaults
            model = request.model or self.config.gemini.model
            temperature = (
                request.temperature
                if request.temperature is not None
                else self.config.gemini.temperature
            )
            max_tokens = (
                request.max_tokens
                if request.max_tokens is not None
                else self.config.gemini.max_tokens
            )
            key = (model, temperature, max_tokens)
            gemini_llm = self._client_cache.get(key)
            if gemini_llm is None:
                gemini_llm = GoogleGenAI(
                    model=model,
                    api_key=self.config.gemini.api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.gemini.timeout,
                )
                self._client_cache[key] = gemini_llm

            # File log (inputs)
            try:
                total_chars = sum(len(m.content or "") for m in request.messages)
                append_line(
                    os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
                    f"gemini_call model={model} temp={temperature} max_tokens={max_tokens} msgs={len(request.messages)} chars={total_chars}",
                )
            except Exception:
                pass

            # Use LlamaIndex's chat method
            response = await gemini_llm.achat(li_messages)

            # Convert LlamaIndex response back to our format
            # Note: LlamaIndex response might not have all fields, so we'll populate what we can
            chat_response = ChatResponse(
                id=f"gemini-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model or self.config.gemini.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(  # type: ignore
                            role=ChatRole.ASSISTANT,
                            content=str(response.message.content),
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatUsage(
                    prompt_tokens=0,  # LlamaIndex may not provide token counts
                    completion_tokens=0,
                    total_tokens=0,
                ),
                system_fingerprint=None,
            )

            # File log (outputs)
            try:
                out_len = len(chat_response.choices[0].message.content or "")
                append_line(
                    os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
                    f"gemini_resp model={model} status=completed out_chars={out_len}",
                )
            except Exception:
                pass

            return chat_response

        except Exception as e:
            logger.error(f"Error in Gemini chat completion: {str(e)}")
            with contextlib.suppress(Exception):
                append_line(
                    os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
                    f"gemini_error model={request.model or self.config.gemini.model} err={str(e).replace('\n', ' ')[:300]}",
                )
            raise

    async def generate_text_with_response(
        self, request: TextGenerationRequest
    ) -> TextGenerationResponse:
        """Generate text with detailed response using LlamaIndex"""
        start_time = time.time()

        try:
            # Generate text using our generate_text method
            text = await self.generate_text(
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                system_prompt=request.system_prompt,
            )

            # Calculate generation time
            generation_time_ms = (time.time() - start_time) * 1000

            return TextGenerationResponse(
                text=text,
                model=request.model or self.config.gemini.model,
                usage=ChatUsage(
                    prompt_tokens=0,  # Gemini API may not provide exact counts
                    completion_tokens=0,
                    total_tokens=0,
                ),
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            logger.error(f"Error generating text with response: {str(e)}")
            raise

    def get_supported_models(self) -> list[str]:
        """Get list of supported Gemini models"""
        return [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.0-pro",
            "gemini-pro",
            "gemini-pro-vision",
        ]

    async def health_check(self) -> bool:
        """Check if the Gemini LLM provider is healthy"""
        try:
            # Simple test with minimal token usage
            response = await self.generate_text(
                prompt="Hello", max_tokens=5, temperature=0.0
            )

            return response is not None and len(response.strip()) > 0

        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            return False
