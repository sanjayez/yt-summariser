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
        """Build the enhanced chain-of-thought system prompt for dependency analysis and query orchestration"""
        from datetime import datetime
        current_year = datetime.now().year
        
        return f"""You are an expert YouTube search orchestrator that uses chain-of-thought reasoning to analyze complex queries and generate optimized search queries with proper dependency ordering.

CHAIN-OF-THOUGHT REASONING PROCESS:

Step 1: DEPENDENCY ANALYSIS
- Identify if concepts have prerequisite relationships (A must come before B)
- Look for temporal indicators: "then", "after", "before", "first", "but don't know"
- Determine logical order: basics → advanced, requirements → goals, problems → solutions
- Flag knowledge gaps that need to be filled first

Step 2: CONCEPT EXPANSION & DOMAIN DETECTION
- Break down broad concepts into specific, searchable topics
- Identify the domain (tech, travel, health, business, science, creative, finance)
- Preserve original context and user intent
- Generate 2-4 related concepts that comprehensively cover the user's needs

Step 3: INTENT CLASSIFICATION
🔍 LOOKUP - Definitions, basic information, "what is" questions
📚 TUTORIAL - Step-by-step learning, complete processes
🛠️ HOW_TO - Specific problem-solving, troubleshooting tasks  
⭐ REVIEW - Comparisons, evaluations, "which is better"
🗺️ GUIDE - Planning, strategies, comprehensive overviews

Step 4: QUERY OPTIMIZATION & ORDERING
- Create specific, actionable YouTube search queries
- Order queries by logical dependency (prerequisites FIRST, then advanced topics)
- Use domain-appropriate keywords and skill-level indicators
- ALWAYS include "english" for language preference
- Ensure queries are specific enough to find quality content

CRITICAL: Show your reasoning process, then provide the JSON.

DETAILED EXAMPLES WITH REASONING:

Example 1 - Complex Travel Query:
Query: "I want to explore the US but I'm not sure where to begin and nor do I know how to apply for visa"

REASONING:
- Domain: Travel planning
- Dependencies: VISA (prerequisite) → TRAVEL PLANNING → DESTINATIONS
- User gaps: visa process knowledge, travel basics, destination ideas  
- Logical order: Can't travel without visa, so visa comes first
- Intent: Comprehensive planning guide

→ {{"intent": "GUIDE", "concepts": ["US visa application process", "US travel planning basics", "best US destinations for tourists"], "enhanced_queries": ["how to apply for US tourist visa step by step english", "US travel planning guide first time international visitors english", "best places to visit in United States tourists english"]}}

Example 2 - Technology Dependency:
Query: "learn React and deploy to production with CI/CD"

REASONING:
- Domain: Web development  
- Dependencies: REACT BASICS → DEPLOYMENT → CI/CD
- User needs: React fundamentals, deployment process, automation
- Logical order: Must know React before deploying React apps
- Intent: Complete learning path

→ {{"intent": "TUTORIAL", "concepts": ["React fundamentals", "React deployment production", "CI/CD for React applications"], "enhanced_queries": ["React tutorial complete beginner javascript english", "deploy React app production hosting english", "CI/CD pipeline React applications english"]}}

Example 3 - Health Prerequisites:
Query: "diabetes diet and safe exercise but I'm newly diagnosed"

REASONING:
- Domain: Health management
- Dependencies: DIABETES BASICS → DIET → EXERCISE
- User gaps: newly diagnosed, needs foundational knowledge first
- Logical order: Understanding condition → diet management → safe exercise
- Intent: Comprehensive health management

→ {{"intent": "GUIDE", "concepts": ["diabetes basics for beginners", "diabetes diet management", "safe exercise with diabetes"], "enhanced_queries": ["diabetes explained newly diagnosed english", "diabetes diet plan beginners english", "safe exercise routines diabetes patients english"]}}

Example 4 - Business Sequence:
Query: "startup idea validation and funding but haven't written business plan"

REASONING:  
- Domain: Business/entrepreneurship
- Dependencies: IDEA VALIDATION → BUSINESS PLAN → FUNDING
- User gaps: validation process, business plan creation, funding knowledge
- Logical order: Validate first, then plan, then seek funding
- Intent: Complete startup process

→ {{"intent": "HOW_TO", "concepts": ["startup idea validation", "business plan creation", "startup funding options"], "enhanced_queries": ["how to validate startup idea market research english", "how to write business plan startups english", "startup funding options investors english"]}}

FORMATTING REQUIREMENTS:
- First show your REASONING process step by step
- Then provide ONLY valid JSON on the last line
- Use exactly these keys: "intent", "concepts", "enhanced_queries"
- Order enhanced_queries by logical dependency (prerequisites first)
- Ensure 2-4 concepts and 2-4 enhanced queries
- Always include "english" in queries

REQUIRED RESPONSE FORMAT:
REASONING:
[Your step-by-step analysis here]

{{"intent": "INTENT_TYPE", "concepts": ["concept1", "concept2", "concept3"], "enhanced_queries": ["prerequisite query english", "main topic query english", "advanced query english"]}}"""

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
            
            # Use LLMService to enhance the query with retry logic
            logger.info(f"Sending query to LLM: '{user_query}'")
            logger.debug(f"Using model: {chat_request.model}")
            
            # DEBUG: Log the system prompt being sent
            logger.debug(f"SYSTEM PROMPT BEING SENT:\n{self.system_prompt}")
            
            # DEBUG: Log the complete chat request details
            logger.debug(f"CHAT REQUEST DETAILS:")
            logger.debug(f"  Model: {chat_request.model}")
            logger.debug(f"  Temperature: {chat_request.temperature}")
            logger.debug(f"  Max tokens: {chat_request.max_tokens}")
            logger.debug(f"  Messages: {[{'role': msg.role.value, 'content': msg.content[:100] + '...' if len(msg.content) > 100 else msg.content} for msg in chat_request.messages]}")
            
            result = await self._llm_with_retry(chat_request, job_id, max_retries=2)
            
            if result["status"] == "completed":
                llm_response = result["response"].choices[0].message.content.strip()
                logger.info(f"LLM raw response: {llm_response}")
                
                # DEBUG: Detailed logging of LLM response
                logger.debug(f"LLM RESPONSE DETAILS:")
                logger.debug(f"  Response length: {len(llm_response)} characters")
                logger.debug(f"  Response starts with: {llm_response[:100]}...")
                logger.debug(f"  Response ends with: ...{llm_response[-100:]}")
                logger.debug(f"  Full response:\n{llm_response}")
                
                # Log processing metrics
                processing_time = result.get("processing_time_ms", 0)
                tokens_used = result["response"].usage.total_tokens
                logger.info(f"Query processing metrics - Time: {processing_time}ms, Tokens: {tokens_used}")
                
                # Parse JSON response containing intent, concepts, and enhanced queries
                try:
                    # Clean the response - sometimes LLM adds extra text before/after JSON
                    logger.debug("Starting JSON extraction from LLM response...")
                    cleaned_response = self._extract_json_from_response(llm_response)
                    logger.debug(f"Cleaned JSON response: {cleaned_response}")
                    
                    if not cleaned_response:
                        logger.error("No valid JSON found in response - cleaned_response is empty")
                        raise json.JSONDecodeError("No valid JSON found in response", llm_response, 0)
                    
                    logger.debug("Attempting to parse JSON...")
                    parsed_response = json.loads(cleaned_response)
                    logger.debug(f"Successfully parsed JSON: {parsed_response}")
                    
                    # Validate required keys
                    required_keys = ["intent", "concepts", "enhanced_queries"]
                    for key in required_keys:
                        if key not in parsed_response:
                            raise KeyError(f"Missing required key: {key}")
                    
                    intent_type = parsed_response.get("intent", "LOOKUP")
                    concepts = parsed_response.get("concepts", [])
                    enhanced_queries = parsed_response.get("enhanced_queries", [])
                    
                    # Validate data types
                    if not isinstance(concepts, list):
                        concepts = [str(concepts)] if concepts else []
                    if not isinstance(enhanced_queries, list):
                        enhanced_queries = [str(enhanced_queries)] if enhanced_queries else []
                    
                    logger.info(f"Parsed response - Intent: {intent_type}, Concepts: {concepts}, Enhanced: {enhanced_queries}")
                    
                    # Log query complexity analysis
                    query_length = len(user_query.split())
                    num_concepts = len(concepts)
                    num_enhanced = len(enhanced_queries)
                    logger.info(f"Query complexity - Words: {query_length}, Concepts: {num_concepts}, Enhanced queries: {num_enhanced}")
                    
                    # Validate extracted data quality
                    if not enhanced_queries or all(not q.strip() for q in enhanced_queries):
                        logger.warning(f"Enhanced queries list is empty or invalid for input: {user_query}")
                        raise ValueError("Empty enhanced queries")
                    
                    if not concepts or all(not c.strip() for c in concepts):
                        logger.warning(f"Concepts list is empty or invalid for input: {user_query}")
                        concepts = [user_query.strip()]
                    
                    # Validate intent type
                    valid_intents = ["LOOKUP", "TUTORIAL", "REVIEW", "HOW_TO", "GUIDE"]
                    if intent_type not in valid_intents:
                        logger.warning(f"Invalid intent '{intent_type}' for query: {user_query}. Defaulting to LOOKUP")
                        intent_type = "LOOKUP"
                    
                    # Clean and validate enhanced queries
                    enhanced_queries = [q.strip() for q in enhanced_queries if q.strip()]
                    if not enhanced_queries:
                        raise ValueError("No valid enhanced queries after cleaning")
                    
                    # Determine if it's a complex query
                    is_complex = len(enhanced_queries) > 1
                    logger.debug(f"Query classified as {'complex' if is_complex else 'simple'}")
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"JSON parse/validation error for query '{user_query}': {e}")
                    logger.error(f"Raw LLM response that failed to parse: {llm_response}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Exception details: {str(e)}")
                    logger.warning(f"Using intelligent fallback analysis.")
                    
                    # DEBUG: Log fallback trigger
                    logger.debug("FALLBACK TRIGGERED - LLM response parsing failed")
                    logger.debug(f"  Error type: {type(e).__name__}")
                    logger.debug(f"  Error message: {str(e)}")
                    logger.debug(f"  Starting fallback analysis for query: '{user_query}'")
                    
                    # Intelligent fallback with semantic analysis
                    concepts, enhanced_queries, intent_type, is_complex = self._intelligent_fallback_analysis(user_query)
                    
                    logger.debug(f"FALLBACK RESULTS:")
                    logger.debug(f"  Concepts: {concepts}")
                    logger.debug(f"  Enhanced queries: {enhanced_queries}")
                    logger.debug(f"  Intent type: {intent_type}")
                    logger.debug(f"  Is complex: {is_complex}")
                
                return {
                    "original_query": user_query.strip(),
                    "concepts": concepts,
                    "enhanced_queries": enhanced_queries,
                    "intent_type": intent_type,
                    "is_complex": is_complex,
                    "job_id": job_id,
                    "status": "completed",
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "tokens_used": result["response"].usage.total_tokens
                }
            else:
                # LLM service failed
                logger.error(f"LLM service failed with status '{result['status']}' for query enhancement: {result.get('error', 'Unknown error')}")
                logger.error(f"Full result: {result}")
                
                # DEBUG: Log LLM service failure details
                logger.debug("LLM SERVICE FAILURE - DETAILED DEBUG:")
                logger.debug(f"  Result status: {result.get('status', 'Unknown')}")
                logger.debug(f"  Error message: {result.get('error', 'No error message provided')}")
                logger.debug(f"  Job ID: {result.get('job_id', 'No job ID')}")
                logger.debug(f"  Full result keys: {list(result.keys())}")
                logger.debug(f"  User query that failed: '{user_query}'")
                logger.debug("  Starting LLM service failure fallback...")
                
                # Intelligent fallback when LLM service fails
                concepts, enhanced_queries, intent_type, is_complex = self._intelligent_fallback_analysis(user_query)
                
                logger.debug(f"LLM SERVICE FAILURE FALLBACK RESULTS:")
                logger.debug(f"  Concepts: {concepts}")
                logger.debug(f"  Enhanced queries: {enhanced_queries}")
                logger.debug(f"  Intent type: {intent_type}")
                logger.debug(f"  Is complex: {is_complex}")
                
                return {
                    "original_query": user_query.strip(),
                    "concepts": concepts,
                    "enhanced_queries": enhanced_queries,
                    "intent_type": intent_type,
                    "is_complex": is_complex,
                    "job_id": job_id,
                    "status": "completed",  # Mark as completed since fallback worked
                    "processing_time_ms": 0,
                    "tokens_used": 0,
                    "fallback_used": True
                }
                
        except Exception as e:
            logger.error(f"Error in query enhancement: {str(e)}")
            return {
                "original_query": user_query.strip() if user_query else "",
                "concepts": [user_query.strip()] if user_query else [],
                "enhanced_queries": [user_query.strip()] if user_query else [],
                "intent_type": "LOOKUP",  # Default intent on exception
                "is_complex": False,
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
        
        # Smart model selection based on query complexity
        model = self._select_model_for_query(user_query)
        logger.debug(f"Selected model '{model}' for query: '{user_query}'")
        
        return ChatRequest(
            messages=messages,
            model=model,
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=500,  # Increased for reasoning + complete JSON response
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    def _select_model_for_query(self, user_query: str) -> str:
        """
        Select the appropriate model based on query complexity and reliability
        
        Args:
            user_query: The user query to analyze
            
        Returns:
            Model name to use
        """
        query_lower = user_query.lower()
        word_count = len(user_query.split())
        
        # Complex query indicators
        has_multiple_concepts = any(connector in query_lower for connector in [" and ", " then ", " plus ", " with ", " or "])
        has_comparison = " vs " in query_lower or "compare" in query_lower
        has_progressive_learning = any(term in query_lower for term in ["basics then", "beginner to", "from scratch"])
        is_architectural = any(term in query_lower for term in ["architecture", "best practices", "design patterns", "microservices"])
        has_dependencies = any(indicator in query_lower for indicator in ["but don't know", "but not sure", "then", "after", "before"])
        
        # For complex dependency analysis, use GPT-4 for better reliability
        if (has_dependencies or 
            has_progressive_learning or
            (has_multiple_concepts and word_count > 15)):
            return "gpt-4o"  # More reliable for complex reasoning
        
        # Use o1-preview for very complex analytical queries
        if (word_count > 20 or 
            is_architectural or
            (has_comparison and word_count > 12)):
            return "o1-preview"
        
        # Use GPT-4 for medium complexity (better JSON consistency)
        if (word_count > 10 or has_multiple_concepts):
            return "gpt-4o"
        
        # Default to o1-mini for simple queries
        return "o1-mini"
    
    def _intelligent_fallback_analysis(self, user_query: str) -> tuple[list[str], list[str], str, bool]:
        """
        Intelligent fallback analysis when LLM fails
        Uses semantic patterns and domain knowledge to extract concepts and generate queries
        
        Args:
            user_query: The original user query
            
        Returns:
            Tuple of (concepts, enhanced_queries, intent_type, is_complex)
        """
        query_lower = user_query.lower().strip()
        
        # Domain detection patterns
        domain_patterns = {
            'technology': ['python', 'javascript', 'react', 'node', 'docker', 'api', 'database', 'programming', 'coding', 'development', 'framework', 'library', 'tutorial', 'coding'],
            'travel': ['travel', 'trip', 'vacation', 'visa', 'flight', 'hotel', 'country', 'city', 'destination', 'tourism', 'explore', 'visit', 'journey'],
            'health': ['health', 'diet', 'exercise', 'nutrition', 'fitness', 'medical', 'doctor', 'treatment', 'symptoms', 'wellness', 'medicine'],
            'business': ['business', 'startup', 'marketing', 'sales', 'finance', 'investment', 'entrepreneur', 'company', 'strategy', 'management'],
            'education': ['learn', 'study', 'course', 'university', 'school', 'education', 'training', 'skill', 'knowledge', 'teach'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'research', 'experiment', 'theory', 'discovery', 'scientific'],
            'creative': ['art', 'design', 'photography', 'music', 'creative', 'drawing', 'painting', 'video', 'editing', 'graphics'],
            'finance': ['money', 'investment', 'crypto', 'stock', 'trading', 'banking', 'finance', 'budget', 'saving', 'loan']
        }
        
        # Detect primary domain
        detected_domain = 'general'
        max_matches = 0
        for domain, keywords in domain_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_domain = domain
        
        # Intent detection patterns
        intent_patterns = {
            'LOOKUP': ['what is', 'what are', 'define', 'explain', 'meaning', 'definition', 'understand', 'overview'],
            'TUTORIAL': ['learn', 'how to', 'tutorial', 'guide', 'step by step', 'course', 'training', 'teach me'],
            'HOW_TO': ['how to', 'fix', 'solve', 'troubleshoot', 'problem', 'error', 'issue', 'repair', 'debug'],
            'REVIEW': ['best', 'vs', 'compare', 'comparison', 'review', 'evaluation', 'which', 'better', 'recommend'],
            'GUIDE': ['guide', 'planning', 'strategy', 'approach', 'process', 'methodology', 'complete guide', 'comprehensive']
        }
        
        # Detect intent
        intent_type = 'LOOKUP'  # default
        max_intent_score = 0
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > max_intent_score:
                max_intent_score = score
                intent_type = intent
        
        # Concept extraction using various delimiters
        concept_delimiters = [' and ', ' then ', ' plus ', ' with ', ' or ', ', ', ' & ']
        concepts = [user_query.strip()]
        
        # Try to split on delimiters
        for delimiter in concept_delimiters:
            if delimiter in query_lower:
                parts = [part.strip() for part in user_query.split(delimiter) if part.strip()]
                if len(parts) > 1:
                    concepts = parts
                    break
        
        # Generate enhanced queries based on domain and intent
        enhanced_queries = []
        for concept in concepts:
            enhanced_query = self._generate_enhanced_query(concept, detected_domain, intent_type)
            enhanced_queries.append(enhanced_query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_enhanced_queries = []
        for query in enhanced_queries:
            if query not in seen:
                seen.add(query)
                unique_enhanced_queries.append(query)
        
        is_complex = len(concepts) > 1
        
        logger.info(f"Fallback analysis - Domain: {detected_domain}, Intent: {intent_type}, Concepts: {len(concepts)}, Complex: {is_complex}")
        
        return concepts, unique_enhanced_queries, intent_type, is_complex
    
    def _generate_enhanced_query(self, concept: str, domain: str, intent: str) -> str:
        """
        Generate an enhanced search query for a specific concept
        
        Args:
            concept: The concept to enhance
            domain: Detected domain
            intent: Detected intent
            
        Returns:
            Enhanced search query
        """
        concept = concept.strip()
        
        # Domain-specific enhancements
        domain_modifiers = {
            'technology': 'tutorial',
            'travel': 'guide',
            'health': 'explained',
            'business': 'guide',
            'education': 'course',
            'science': 'explained',
            'creative': 'tutorial',
            'finance': 'explained'
        }
        
        # Intent-specific modifiers
        intent_modifiers = {
            'LOOKUP': 'explained',
            'TUTORIAL': 'tutorial',
            'HOW_TO': 'guide',
            'REVIEW': 'review',
            'GUIDE': 'complete guide'
        }
        
        # Start with the concept
        enhanced = concept
        
        # Add intent modifier
        modifier = intent_modifiers.get(intent, domain_modifiers.get(domain, 'guide'))
        if modifier not in enhanced.lower():
            enhanced += f" {modifier}"
        
        # Add skill level if not present
        skill_indicators = ['beginner', 'advanced', 'complete', 'basic', 'intro']
        if not any(indicator in enhanced.lower() for indicator in skill_indicators):
            if intent == 'LOOKUP':
                enhanced += " beginner"
            elif domain == 'technology':
                enhanced += " complete"
        
        # Always add english for language preference
        if 'english' not in enhanced.lower():
            enhanced += " english"
        
        return enhanced
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response that might contain extra text
        
        Args:
            response: Raw LLM response
            
        Returns:
            Clean JSON string or empty string if not found
        """
        if not response or not response.strip():
            return ""
        
        response = response.strip()
        
        # Try to find JSON block between curly braces
        import re
        
        # First try to find JSON after "REASONING:" section
        if "REASONING:" in response:
            # Split by REASONING and take everything after it
            parts = response.split("REASONING:", 1)
            if len(parts) > 1:
                after_reasoning = parts[1].strip()
                # Look for JSON in the reasoning section
                json_match = re.search(r'\{[\s\S]*?"intent"[\s\S]*?"concepts"[\s\S]*?"enhanced_queries"[\s\S]*?\}', after_reasoning, re.DOTALL)
                if json_match:
                    return json_match.group(0).strip()
        
        # Look for JSON patterns anywhere in response
        json_patterns = [
            r'\{[^{}]*"intent"[^{}]*"concepts"[^{}]*"enhanced_queries"[^{}]*\}',  # Single line JSON
            r'\{[\s\S]*?"intent"[\s\S]*?"concepts"[\s\S]*?"enhanced_queries"[\s\S]*?\}',  # Multi-line JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the first match
                return matches[0].strip()
        
        # If no JSON pattern found, try to extract everything between first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = response[first_brace:last_brace + 1]
            # Basic validation - check if it contains our required keys
            if '"intent"' in potential_json and '"concepts"' in potential_json and '"enhanced_queries"' in potential_json:
                return potential_json.strip()
        
        # Handle truncated JSON - try to complete it
        if first_brace != -1 and '"intent"' in response and '"concepts"' in response and '"enhanced_queries"' in response:
            potential_json = response[first_brace:]
            # If it looks like truncated JSON, try to complete it
            if not potential_json.endswith('}') and not potential_json.endswith(']}'):
                if potential_json.endswith('"'):
                    potential_json += ']}'
                elif potential_json.endswith(']'):
                    potential_json += '}'
                else:
                    potential_json += '"]}'
                
            # Validate if it's now proper JSON
            try:
                import json
                json.loads(potential_json)
                return potential_json.strip()
            except json.JSONDecodeError:
                pass
        
        # Last resort - return the original response if it looks like JSON
        if response.startswith('{') and response.endswith('}'):
            return response
        
        return ""
    
    async def _llm_with_retry(self, chat_request: ChatRequest, job_id: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Call LLM with retry logic for better reliability
        
        Args:
            chat_request: The chat request to send
            job_id: Job identifier  
            max_retries: Maximum number of retries
            
        Returns:
            LLM service result
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await self.llm_service.chat_completion(chat_request, job_id=f"{job_id}_attempt_{attempt}")
                
                # Check if result is valid
                if result["status"] == "completed":
                    response_content = result["response"].choices[0].message.content
                    if response_content and response_content.strip():
                        logger.debug(f"LLM call succeeded on attempt {attempt + 1}")
                        return result
                    else:
                        logger.warning(f"LLM returned empty response on attempt {attempt + 1}")
                        last_error = "Empty LLM response"
                else:
                    logger.warning(f"LLM call failed on attempt {attempt + 1}: {result.get('error', 'Unknown error')}")
                    last_error = result.get('error', 'Unknown error')
                
                # If this was the last attempt, return the result anyway
                if attempt == max_retries:
                    return result
                    
                # Try different model on retry for better success rate
                if attempt == 0 and chat_request.model.startswith("o1"):
                    logger.info(f"Retrying with GPT-4 instead of {chat_request.model}")
                    chat_request.model = "gpt-4o"
                elif attempt == 1 and chat_request.model == "gpt-4o":
                    logger.info(f"Retrying with o1-mini as final attempt")
                    chat_request.model = "o1-mini"
                    
            except Exception as e:
                logger.error(f"LLM call exception on attempt {attempt + 1}: {str(e)}")
                last_error = str(e)
                
                if attempt == max_retries:
                    # Return failed result on final attempt
                    return {
                        "status": "failed",
                        "error": f"LLM failed after {max_retries + 1} attempts: {last_error}",
                        "job_id": job_id
                    }
        
        # Should not reach here, but just in case
        return {
            "status": "failed", 
            "error": f"LLM retry logic failed: {last_error}",
            "job_id": job_id
        }
    
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
                        "concepts": result.get("concepts", [query.strip()]),
                        "enhanced_queries": result.get("enhanced_queries", [query.strip()]),
                        "intent_type": result.get("intent_type", "LOOKUP"),
                        "is_complex": result.get("is_complex", False),
                        "status": result.get("status", "failed"),
                        "processing_time_ms": result.get("processing_time_ms", 0),
                        "tokens_used": result.get("tokens_used", 0)
                    })
                except Exception as e:
                    logger.error(f"Error processing query {i}: {str(e)}")
                    results.append({
                        "index": i,
                        "original_query": query.strip(),
                        "concepts": [query.strip()],
                        "enhanced_queries": [query.strip()],
                        "intent_type": "LOOKUP",
                        "is_complex": False,
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