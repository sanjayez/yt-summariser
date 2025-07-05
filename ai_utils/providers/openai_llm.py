"""
OpenAI LLM Provider Implementation
Handles chat completions and text generation using OpenAI's API
"""

import time
import asyncio
import logging
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from ..interfaces.llm import LLMProvider
from ..models import (
    RAGQuery, RAGResponse, VectorSearchResult, 
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ChatChoice, ChatUsage,
    TextGenerationRequest, TextGenerationResponse
)
from ..config import AIConfig

logger = logging.getLogger(__name__)

class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider for chat completions and text generation"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
            timeout=config.openai.timeout,
            max_retries=config.openai.max_retries
        )
        self.default_model = config.openai.chat_model
        self.default_temperature = config.openai.temperature
        self.default_max_tokens = config.openai.max_tokens
        self.default_top_p = config.openai.top_p
        self.default_frequency_penalty = config.openai.frequency_penalty
        self.default_presence_penalty = config.openai.presence_penalty
        self.default_user = config.openai.user
    
    async def generate_rag_response(
        self, 
        query: RAGQuery, 
        context_documents: List[VectorSearchResult]
    ) -> RAGResponse:
        """Generate a RAG response using context documents"""
        start_time = time.time()
        
        try:
            # Build context from documents
            context = self._build_context(context_documents)
            
            # Create system prompt for RAG
            system_prompt = self._create_rag_system_prompt()
            
            # Create user prompt with context and query
            user_prompt = self._create_rag_user_prompt(query.query, context)
            
            # Prepare chat messages
            messages = [
                ChatMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMessage(role=ChatRole.USER, content=user_prompt)
            ]
            
            # Generate response
            response = await self._chat_completion(
                messages=messages,
                model=self.default_model,
                temperature=query.temperature,
                max_tokens=query.max_tokens
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            return RAGResponse(
                answer=answer,
                sources=context_documents,
                query=query.query,
                response_time_ms=response_time_ms,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            raise
    
    async def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from a prompt"""
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system_prompt))
            messages.append(ChatMessage(role=ChatRole.USER, content=prompt))
            
            # Generate response
            response = await self._chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Complete chat conversation"""
        try:
            response = await self._chat_completion(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                stream=request.stream,
                user=request.user
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    async def generate_text_with_response(
        self, 
        request: TextGenerationRequest
    ) -> TextGenerationResponse:
        """Generate text with detailed response"""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append(ChatMessage(role=ChatRole.SYSTEM, content=request.system_prompt))
            messages.append(ChatMessage(role=ChatRole.USER, content=request.prompt))
            
            # Generate response
            response = await self._chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Calculate generation time
            generation_time_ms = (time.time() - start_time) * 1000
            
            return TextGenerationResponse(
                text=response.choices[0].message.content,
                model=response.model,
                usage=response.usage,
                generation_time_ms=generation_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error generating text with response: {str(e)}")
            raise
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported chat models"""
        return [
            "gpt-4",
            "gpt-4-1106-preview",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613"
        ]
    
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy"""
        try:
            # Simple test with minimal token usage
            messages = [ChatMessage(role=ChatRole.USER, content="Hello")]
            response = await self._chat_completion(
                messages=messages,
                model=self.default_model,
                max_tokens=1,
                temperature=0.0
            )
            
            return response.choices[0].message.content is not None
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def _chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        user: Optional[str] = None
    ) -> ChatResponse:
        """Internal method for chat completions"""
        
        # Prepare OpenAI API call parameters
        params = {
            "model": model or self.default_model,
            "messages": [{"role": msg.role.value, "content": msg.content} for msg in messages],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "top_p": top_p if top_p is not None else self.default_top_p,
            "frequency_penalty": frequency_penalty if frequency_penalty is not None else self.default_frequency_penalty,
            "presence_penalty": presence_penalty if presence_penalty is not None else self.default_presence_penalty,
            "stream": stream,
            "user": user or self.default_user
        }
        
        # Add optional parameters
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        elif self.default_max_tokens is not None:
            params["max_tokens"] = self.default_max_tokens
            
        if stop is not None:
            params["stop"] = stop
        
        # Make API call
        response = await self.client.chat.completions.create(**params)
        
        # Convert to our ChatResponse model
        return ChatResponse(
            id=response.id,
            object=response.object,
            created=response.created,
            model=response.model,
            choices=[
                ChatChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=ChatRole(choice.message.role),
                        content=choice.message.content
                    ),
                    finish_reason=choice.finish_reason
                ) for choice in response.choices
            ],
            usage=ChatUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ),
            system_fingerprint=getattr(response, 'system_fingerprint', None)
        )
    
    def _build_context(self, documents: List[VectorSearchResult]) -> str:
        """Build context string from search results"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:")
            context_parts.append(f"Content: {doc.text}")
            if doc.metadata:
                context_parts.append(f"Metadata: {doc.metadata}")
            context_parts.append("")  # Empty line for readability
        
        return "\n".join(context_parts)
    
    def _create_rag_system_prompt(self) -> str:
        """Create system prompt for RAG responses"""
        return """You are a helpful assistant that answers questions based on provided context documents.

Instructions:
1. Use only the information provided in the context documents to answer questions
2. If the context doesn't contain enough information to answer the question, clearly state that
3. Be concise and accurate in your responses
4. If you reference specific information, indicate which document it came from
5. Do not make up information that isn't in the context

Your goal is to provide helpful, accurate answers based solely on the provided context."""
    
    def _create_rag_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt for RAG with context and query"""
        return f"""Context Documents:
{context}

Question: {query}

Please answer the question based on the provided context documents.""" 