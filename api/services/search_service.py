"""
Search Service for RAG operations and video content search.
This service breaks down the massive search_video_content_async function into manageable components.
"""
from typing import Dict, List, Any, Optional, Tuple
from telemetry import get_logger, timed_operation, handle_exceptions, TimingContext
from .service_container import get_service_container
from .cache_service import CacheService
from .response_service import ResponseService


class SearchService:
    """
    Handles RAG search operations for video content.
    
    This service breaks down the original 210-line search_video_content_async function
    into focused, testable methods following SOLID principles.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.service_container = get_service_container()
        self.cache_service = CacheService()
        self.response_service = ResponseService()
    
    @handle_exceptions(reraise=True)
    @timed_operation()
    async def search_video_content(self, question: str, video_metadata, transcript) -> Dict[str, Any]:
        """
        Main search orchestration method.
        
        This replaces the original 210-line search_video_content_async function
        with a clean, focused approach using proper async patterns.
        
        Args:
            question: User's question about the video
            video_metadata: VideoMetadata model instance
            transcript: VideoTranscript model instance
            
        Returns:
            Dictionary with search results, sources, and metadata
        """
        stage_timings = {}
        video_id = video_metadata.video_id
        
        try:
            # Stage 1: Service Initialization
            with TimingContext("service_initialization") as timer:
                services = await self._initialize_services()
            stage_timings['1_SERVICE_INIT'] = timer.elapsed_ms
            
            # Stage 2: Vector Search
            with TimingContext("vector_search") as timer:
                search_results = await self._perform_vector_search(
                    question, video_id, services
                )
            stage_timings['2_VECTOR_SEARCH_TOTAL'] = timer.elapsed_ms
            
            # Stage 3: Context Processing
            with TimingContext("context_processing") as timer:
                processed_context, formatted_sources = await self._process_search_results(
                    search_results, video_metadata
                )
            stage_timings['3_CONTEXT_PROCESSING'] = timer.elapsed_ms
            
            # Stage 4: Answer Generation
            if processed_context and processed_context.strip():
                with TimingContext("answer_generation") as timer:
                    answer = await self._generate_answer(
                        question, processed_context, services
                    )
                stage_timings['4_ANSWER_GENERATION'] = timer.elapsed_ms
            else:
                answer = "I couldn't find relevant information in the video to answer your question."
                stage_timings['4_ANSWER_GENERATION'] = 0
            
            # Stage 5: Response Assembly
            with TimingContext("response_assembly") as timer:
                response = await self._assemble_response(
                    answer, formatted_sources, search_results, stage_timings
                )
            stage_timings['5_RESPONSE_ASSEMBLY'] = timer.elapsed_ms
            
            return response
            
        except Exception as e:
            self.logger.error(f"Vector search failed for video {video_id}: {e}")
            
            # Log partial timing data
            if stage_timings:
                self.logger.error(f"RAG pipeline error after {len(stage_timings)} stages")
            
            raise
    
    async def _initialize_services(self) -> Dict[str, Any]:
        """
        Initialize required services for search operations.
        
        Returns:
            Dictionary with initialized service instances
        """
        try:
            services = self.service_container.get_ai_services()
            self.logger.debug("Services initialized for search operation")
            return services
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            raise
    
    async def _perform_vector_search(self, question: str, video_id: str, services: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform vector search for the question against video content.
        
        Args:
            question: User's question
            video_id: Video identifier
            services: Initialized AI services
            
        Returns:
            List of search results
        """
        search_results = []
        
        try:
            self.logger.info(f"🔍 Starting vector search for video {video_id} with question: '{question}'")
            
            vector_service = services['vector']
            segments_results = await vector_service.search_by_text(
                text=question,
                top_k=5,
                filters={'video_id': video_id, 'type': 'segment'}
            )
            
            self.logger.info(f"🔍 Vector search completed. Results: {len(segments_results.results) if segments_results else 0}")
            
            if segments_results and segments_results.results:
                self.logger.info(f"🔍 Processing {len(segments_results.results)} results")
                for i, result in enumerate(segments_results.results):
                    self.logger.info(f"🔍 Result {i+1}: score={result.score:.4f}, text='{result.text[:50]}...'")
                    search_results.append({
                        'type': 'segment',
                        'text': result.text,
                        'score': result.score,
                        'metadata': result.metadata
                    })
            else:
                self.logger.warning(f"🔍 No vector search results found for video {video_id}")
                    
        except Exception as e:
            self.logger.error(f"🔍 Vector search failed: {e}")
            import traceback
            self.logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            # Return empty results rather than failing completely
            
        return search_results
    
    async def _process_search_results(self, search_results: List[Dict[str, Any]], video_metadata) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process search results and prepare context and sources.
        
        Args:
            search_results: Raw search results from vector search
            video_metadata: Video metadata for context
            
        Returns:
            Tuple of (compressed_context, formatted_sources)
        """
        # Sort by relevance score and take top 3
        search_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_results = search_results[:3]
        
        # Use response service for context compression
        compressed_context = self.response_service.compress_context(top_results, max_chars=800)
        
        # Format sources for response
        formatted_sources = self.response_service.format_search_sources(top_results, video_metadata)
        
        return compressed_context, formatted_sources
    
    async def _generate_answer(self, question: str, context: str, services: Dict[str, Any]) -> str:
        """
        Generate answer using LLM with context.
        Uses only Gemini (no OpenAI fallback as requested).
        
        Args:
            question: User's question
            context: Compressed context from search results  
            services: Initialized AI services
            
        Returns:
            Generated answer string
        """
        from ai_utils.models import ChatMessage, ChatRequest
        
        # Optimized prompt using actual context
        prompt = f"""Based on this video content, answer the question concisely:

{context}

Question: {question}
Answer:"""
        
        messages = [ChatMessage(role="user", content=prompt)]
        
        try:
            # Use Gemini only (no OpenAI fallback)
            gemini_request = ChatRequest(
                messages=messages,
                model="gemini-2.5-flash",
                temperature=0.3,
                max_tokens=4000  # High limit to prevent MAX_TOKENS errors
            )
            
            self.logger.info("🔍 Generating answer with Gemini...")
            llm_service = services['llm']
            gemini_response = await llm_service.provider.chat_completion(gemini_request)
            
            if gemini_response and hasattr(gemini_response, 'choices') and gemini_response.choices:
                answer = gemini_response.choices[0].message.content.strip()
                self.logger.info(f"✅ Gemini success: {answer[:100]}...")
                return answer
            else:
                raise Exception("Gemini returned no choices")
                    
        except Exception as gemini_error:
            self.logger.error(f"❌ Gemini failed: {gemini_error}")
            return "I couldn't generate an answer based on the available content."
    
    async def _assemble_response(self, answer: str, sources: List[Dict[str, Any]], 
                                search_results: List[Dict[str, Any]], stage_timings: Dict[str, float]) -> Dict[str, Any]:
        """
        Assemble final response with all components.
        
        Args:
            answer: Generated answer
            sources: Formatted sources
            search_results: Original search results for confidence calculation
            stage_timings: Timing information for each stage
            
        Returns:
            Complete response dictionary
        """
        # Calculate overall confidence
        confidence = self.response_service.calculate_confidence_score(search_results)
        
        response = {
            'answer': answer,
            'sources': sources[:3],  # Top 3 sources
            'confidence': confidence,
            'search_method': 'vector_search',
            'results_count': len(search_results)
        }
        
        # Add comprehensive timing information
        timing_info = self.response_service.format_timing_info(stage_timings)
        if timing_info:
            response['timing'] = timing_info
        
        # Log timing summary for debugging
        total_time = sum(stage_timings.values())
        self.logger.debug(f"RAG pipeline completed in {total_time:.2f}ms with {len(stage_timings)} stages")
        
        return response
    
    @handle_exceptions(reraise=True)
    async def search_transcript_fallback(self, question: str, transcript_text: str, video_metadata) -> Dict[str, Any]:
        """
        Fallback search using raw transcript text when embeddings are not available.
        
        This method was extracted from the original search_transcript_fallback function
        in api/views.py lines 470-536.
        
        Args:
            question: User's question
            transcript_text: Raw transcript text
            video_metadata: Video metadata for context
            
        Returns:
            Search result dictionary with transcript-based results
        """
        try:
            # Simple keyword matching in transcript
            question_words = question.lower().split()
            transcript_lower = transcript_text.lower()
            
            # Find sentences containing question keywords
            sentences = transcript_text.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                matches = sum(1 for word in question_words if word in sentence_lower)
                if matches > 0:
                    relevant_sentences.append({
                        'text': sentence.strip(),
                        'matches': matches,
                        'relevance': matches / len(question_words)
                    })
            
            # Sort by relevance and take top 3
            relevant_sentences.sort(key=lambda x: x['relevance'], reverse=True)
            top_sentences = relevant_sentences[:3]
            
            # Create simple answer
            if top_sentences:
                context = "\n".join([s['text'] for s in top_sentences])
                answer = f"Based on the video transcript, here are the most relevant parts:\n\n{context}"
                
                sources = [{
                    'type': 'transcript',
                    'timestamp': 'Unknown',
                    'text': s['text'],
                    'youtube_url': video_metadata.webpage_url if video_metadata else '',
                    'confidence': s['relevance']
                } for s in top_sentences]
                
                avg_confidence = sum(s['relevance'] for s in top_sentences) / len(top_sentences)
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'confidence': round(avg_confidence, 2),
                    'search_method': 'transcript_fallback',
                    'results_count': len(top_sentences)
                }
            else:
                return {
                    'answer': "I couldn't find relevant information in the transcript to answer your question.",
                    'sources': [],
                    'confidence': 0.0,
                    'search_method': 'transcript_fallback',
                    'results_count': 0
                }
                
        except Exception as e:
            self.logger.error(f"Transcript fallback search failed: {e}")
            return {
                'answer': f"Error searching transcript: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'search_method': 'transcript_fallback',
                'results_count': 0
            }