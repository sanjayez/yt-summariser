"""
QueryProcessor Service
Enhances user queries by converting natural language to effective YouTube search terms
"""

import logging
import json
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
        """Build the system prompt for combined intent classification and query enhancement"""
        from datetime import datetime
        current_year = datetime.now().year
        
        return f"""You are a YouTube search optimizer with intent classification capabilities. For each query, classify the intent and optimize the search terms.

INTENT CATEGORIES (choose ONE):

🔍 LOOKUP - Quick facts, definitions, basic information
   • Markers: "what is", "who is", "define", "meaning", "specs", "age"
   • Examples: "What is machine learning?", "iPhone 15 specs", "Who is Elon Musk?"

📚 TUTORIAL - Step-by-step learning processes  
   • Markers: "learn", "tutorial", "course", "lessons", "for beginners"
   • Examples: "Learn Python programming", "Guitar lessons for beginners"

🛠️ HOW_TO - Solving specific problems/tasks
   • Markers: "how to", "fix", "solve", "repair", "build", "make"
   • Examples: "How to fix iPhone screen", "How to bake bread"

⭐ REVIEW - Opinions, evaluations, comparisons, recommendations
   • Markers: "review", "vs", "comparison", "best", "top", "better"
   • Examples: "iPhone 15 review", "React vs Vue", "Best laptops {current_year}"

CLASSIFICATION RULES:
- LOOKUP: Simple information seeking (definitions, facts, specs)
- TUTORIAL: Structured learning (courses, lessons, skill building)
- HOW_TO: Problem solving (repairs, tasks, specific goals)
- REVIEW: Evaluations (reviews, comparisons, recommendations)

QUERY OPTIMIZATION:
- Convert to specific, searchable keywords
- Remove unnecessary words ("how to", "what is", "explain")  
- Include relevant technical terms and synonyms
- Aim for 3-7 keywords maximum (including "english")
- ALWAYS include "english" to ensure English content
- For "latest" queries, use {current_year} or omit year
- Prioritize terms that appear in YouTube video titles

RESPONSE FORMAT (JSON only):
{{"intent": "INTENT_TYPE", "enhanced_query": "optimized search terms"}}

EXAMPLES:
- "What is machine learning?" → {{"intent": "LOOKUP", "enhanced_query": "machine learning definition explained english"}}
- "Learn Python programming" → {{"intent": "TUTORIAL", "enhanced_query": "Python programming tutorial course beginner english"}}
- "How to fix iPhone screen" → {{"intent": "HOW_TO", "enhanced_query": "iPhone screen repair guide english"}}
- "iPhone 15 vs Samsung Galaxy review" → {{"intent": "REVIEW", "enhanced_query": "iPhone 15 Samsung Galaxy comparison review {current_year} english"}}
- "Best laptops 2024" → {{"intent": "REVIEW", "enhanced_query": "best laptops {current_year} review comparison english"}}
- "JavaScript course for beginners" → {{"intent": "TUTORIAL", "enhanced_query": "JavaScript course tutorial beginner english"}}

Return ONLY valid JSON, nothing else."""

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
                llm_response = result["response"].choices[0].message.content.strip()
                
                # Parse JSON response containing intent and enhanced query
                try:
                    parsed_response = json.loads(llm_response)
                    intent_type = parsed_response.get("intent", "FACTUAL")
                    enhanced_query = parsed_response.get("enhanced_query", "")
                    
                    # Validate extracted data
                    if not enhanced_query:
                        logger.warning(f"Enhanced query is empty for input: {user_query}")
                        enhanced_query = user_query.strip()
                    
                    # Validate intent type
                    valid_intents = ["LOOKUP", "TUTORIAL", "REVIEW", "HOW_TO"]
                    if intent_type not in valid_intents:
                        logger.warning(f"Invalid intent '{intent_type}' for query: {user_query}. Defaulting to LOOKUP")
                        intent_type = "LOOKUP"
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse JSON response for query '{user_query}': {e}. Using fallback values.")
                    enhanced_query = user_query.strip()
                    intent_type = "LOOKUP"
                
                return {
                    "original_query": user_query.strip(),
                    "enhanced_query": enhanced_query,
                    "intent_type": intent_type,
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
                    "intent_type": "LOOKUP",  # Default intent on failure
                    "job_id": job_id,
                    "status": "failed",
                    "error": result.get("error", "Query enhancement failed")
                }
                
        except Exception as e:
            logger.error(f"Error in query enhancement: {str(e)}")
            return {
                "original_query": user_query.strip() if user_query else "",
                "enhanced_query": user_query.strip() if user_query else "",
                "intent_type": "LOOKUP",  # Default intent on exception
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
            max_tokens=150,  # Increased for JSON response with intent and enhanced query
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
                        "intent_type": result.get("intent_type", "LOOKUP"),
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
                        "intent_type": "LOOKUP",
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