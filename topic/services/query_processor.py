"""
QueryProcessor Service
Enhances user queries by converting natural language to effective YouTube search terms
"""

import logging
from typing import Dict, Any, Optional
from uuid import uuid4

from ai_utils.services.llm_service import LLMService
from ai_utils.models import ChatRequest, ChatMessage, ChatRole, ProcessingStatus

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    QueryProcessor service that uses LLMService to enhance user queries
    for better YouTube search results
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize QueryProcessor with LLMService
        
        Args:
            llm_service: Instance of LLMService for AI operations
        """
        self.llm_service = llm_service
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for query enhancement"""
        from datetime import datetime
        current_year = datetime.now().year
        
        return f"""You are a YouTube search query optimizer. Your task is to convert natural language queries into effective YouTube search terms that work well with YouTube's algorithm.

Guidelines:
- Convert conversational queries to specific, searchable keywords
- Include relevant technical terms and synonyms  
- Focus on actionable and specific terms
- Remove unnecessary words like "how to", "explain", "what is"
- Keep essential context and domain-specific terms
- Aim for 3-7 keywords maximum (including "english")
- ALWAYS include "english" as a keyword to ensure English-language content
- DO NOT use quotes around the entire query
- For "latest" or "newest" queries, use {current_year} or omit year entirely
- Prioritize terms that would appear in YouTube video titles
- Use simple space-separated keywords, not quoted phrases

Examples:
- "How do I learn Python programming?" → "Python programming tutorial beginner english"
- "Can you explain machine learning concepts?" → "machine learning concepts explained english"
- "What are the best practices for React development?" → "React development best practices english"
- "I want to understand neural networks" → "neural networks explained tutorial english"
- "What are the latest phones?" → "latest phones {current_year} review english"
- "Show me newest smartphone reviews" → "newest smartphone reviews {current_year} english"

Return only the enhanced search terms, nothing else."""

    async def enhance_query(
        self, 
        user_query: str, 
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance a user query by converting it to effective YouTube search terms
        
        Args:
            user_query: The original user query in natural language
            job_id: Optional job identifier for tracking
            
        Returns:
            Dict containing enhanced query, job status, and processing info
        """
        job_id = job_id or f"query_enhance_{uuid4().hex[:8]}"
        
        try:
            # Validate input
            if not user_query or not user_query.strip():
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": "Query cannot be empty"
                }
            
            # Create chat request for query enhancement
            chat_request = self._create_enhancement_request(user_query.strip())
            
            # Use LLMService to enhance the query
            result = await self.llm_service.chat_completion(chat_request, job_id=job_id)
            
            if result["status"] == "completed":
                enhanced_query = result["response"].choices[0].message.content.strip()
                
                # Basic validation of the enhanced query
                if not enhanced_query:
                    logger.warning(f"Enhanced query is empty for input: {user_query}")
                    enhanced_query = user_query.strip()
                
                return {
                    "original_query": user_query.strip(),
                    "enhanced_query": enhanced_query,
                    "job_id": job_id,
                    "status": "completed",
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "tokens_used": result["response"].usage.total_tokens
                }
            else:
                # LLM service failed
                logger.error(f"LLM service failed for query enhancement: {result.get('error', 'Unknown error')}")
                return {
                    "original_query": user_query.strip(),
                    "enhanced_query": user_query.strip(),  # Fallback to original
                    "job_id": job_id,
                    "status": "failed",
                    "error": result.get("error", "Query enhancement failed")
                }
                
        except Exception as e:
            logger.error(f"Error in query enhancement: {str(e)}")
            return {
                "original_query": user_query.strip() if user_query else "",
                "enhanced_query": user_query.strip() if user_query else "",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _create_enhancement_request(self, user_query: str) -> ChatRequest:
        """
        Create a ChatRequest for query enhancement
        
        Args:
            user_query: The user query to enhance
            
        Returns:
            ChatRequest configured for query enhancement
        """
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt),
            ChatMessage(role=ChatRole.USER, content=user_query)
        ]
        
        return ChatRequest(
            messages=messages,
            model="gpt-3.5-turbo",  # Fast and cost-effective for this task
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=100,  # Short responses expected
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    async def batch_enhance_queries(
        self, 
        queries: list[str], 
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance multiple queries in batch
        
        Args:
            queries: List of user queries to enhance
            job_id: Optional job identifier for tracking
            
        Returns:
            Dict containing batch results and processing info
        """
        job_id = job_id or f"batch_query_enhance_{uuid4().hex[:8]}"
        
        try:
            # Validate input
            if not queries or not all(q.strip() for q in queries):
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": "All queries must be non-empty"
                }
            
            # Process queries concurrently
            results = []
            for i, query in enumerate(queries):
                try:
                    result = await self.enhance_query(query, job_id=f"{job_id}_item_{i}")
                    results.append({
                        "index": i,
                        "original_query": query.strip(),
                        "enhanced_query": result.get("enhanced_query", query.strip()),
                        "status": result.get("status", "failed"),
                        "processing_time_ms": result.get("processing_time_ms", 0),
                        "tokens_used": result.get("tokens_used", 0)
                    })
                except Exception as e:
                    logger.error(f"Error processing query {i}: {str(e)}")
                    results.append({
                        "index": i,
                        "original_query": query.strip(),
                        "enhanced_query": query.strip(),
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Calculate totals
            total_processing_time = sum(r.get("processing_time_ms", 0) for r in results)
            total_tokens = sum(r.get("tokens_used", 0) for r in results)
            successful_count = sum(1 for r in results if r.get("status") == "completed")
            
            return {
                "results": results,
                "job_id": job_id,
                "status": "completed",
                "total_queries": len(queries),
                "successful_enhancements": successful_count,
                "failed_enhancements": len(queries) - successful_count,
                "total_processing_time_ms": total_processing_time,
                "total_tokens_used": total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error in batch query enhancement: {str(e)}")
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check QueryProcessor health by testing LLM service
        
        Returns:
            Dict containing health status and info
        """
        try:
            # Test with a simple query
            test_result = await self.enhance_query("test query", job_id="health_check")
            
            if test_result["status"] == "completed":
                return {
                    "status": "healthy",
                    "service": "QueryProcessor",
                    "llm_service_status": "healthy",
                    "test_processing_time_ms": test_result.get("processing_time_ms", 0)
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": "QueryProcessor",
                    "llm_service_status": "unhealthy",
                    "error": test_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "QueryProcessor",
                "error": str(e)
            }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific job from the underlying LLM service
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information or None if not found
        """
        job = self.llm_service.get_job_status(job_id)
        if job:
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "operation": job.operation,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "error_message": job.error_message
            }
        return None