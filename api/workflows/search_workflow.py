"""
LlamaIndex Workflow for API-specific RAG search operations.
Provides event-driven, observable search with proper async patterns.
"""
from typing import Dict, List, Any, Optional
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context
from llama_index.core.workflow.events import Event
from telemetry import get_logger, TimingContext
from ..services.service_container import get_service_container
from ..services.response_service import ResponseService


class SearchInitEvent(Event):
    """Event to initialize search context"""
    question: str
    video_id: str
    video_metadata: Any
    transcript: Any


class VectorSearchEvent(Event):
    """Event containing vector search results"""
    question: str
    video_id: str
    results: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class ContextProcessedEvent(Event):
    """Event containing processed context and sources"""
    question: str
    compressed_context: str
    formatted_sources: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]


class AnswerGeneratedEvent(Event):
    """Event containing generated answer"""
    question: str
    answer: str
    compressed_context: str
    formatted_sources: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]


class APISearchWorkflow(Workflow):
    """
    Event-driven RAG workflow for video search operations.
    
    This workflow provides a more sophisticated, observable alternative to the
    traditional sequential search approach. It offers:
    - Event-driven architecture for better async handling
    - Built-in observability and tracing
    - Composable and reusable components
    - Proper error handling at each stage
    """
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.service_container = get_service_container()
        self.response_service = ResponseService()
    
    @step
    async def initialize_search(self, ctx: Context, ev: StartEvent) -> SearchInitEvent:
        """
        Initialize search context and validate inputs.
        
        Args:
            ctx: Workflow context
            ev: Start event with search parameters
            
        Returns:
            SearchInitEvent with validated parameters
        """
        async with TimingContext("workflow_initialization") as timer:
            # Validate inputs
            question = ev.question.strip()
            if not question:
                raise ValueError("Question cannot be empty")
            
            video_id = ev.video_metadata.video_id
            self.logger.info(f"🔄 Initializing search workflow for video {video_id}")
            
            # Store timing in context
            ctx.data.setdefault("timings", {})["initialization"] = timer.elapsed_ms
            
        return SearchInitEvent(
            question=question,
            video_id=video_id,
            video_metadata=ev.video_metadata,
            transcript=ev.transcript
        )
    
    @step
    async def perform_vector_search(self, ctx: Context, ev: SearchInitEvent) -> VectorSearchEvent:
        """
        Perform vector search against video content.
        
        Args:
            ctx: Workflow context
            ev: Search initialization event
            
        Returns:
            VectorSearchEvent with search results
        """
        async with TimingContext("vector_search") as timer:
            try:
                # Initialize AI services
                services = self.service_container.get_ai_services()
                vector_service = services['vector']
                
                self.logger.info(f"🔍 Performing vector search: '{ev.question[:50]}...'")
                
                # Execute vector search
                segments_results = await vector_service.search_by_text(
                    text=ev.question,
                    top_k=5,
                    filters={'video_id': ev.video_id, 'type': 'segment'}
                )
                
                results = []
                if segments_results and segments_results.results:
                    self.logger.info(f"🔍 Found {len(segments_results.results)} vector results")
                    for result in segments_results.results:
                        results.append({
                            'type': 'segment',
                            'text': result.text,
                            'score': result.score,
                            'metadata': result.metadata
                        })
                else:
                    self.logger.warning(f"🔍 No vector results found for video {ev.video_id}")
                
                # Store timing
                ctx.data["timings"]["vector_search"] = timer.elapsed_ms
                
                return VectorSearchEvent(
                    question=ev.question,
                    video_id=ev.video_id,
                    results=results,
                    success=True
                )
                
            except Exception as e:
                self.logger.error(f"🔍 Vector search failed: {e}")
                ctx.data["timings"]["vector_search"] = timer.elapsed_ms
                
                return VectorSearchEvent(
                    question=ev.question,
                    video_id=ev.video_id,
                    results=[],
                    success=False,
                    error=str(e)
                )
    
    @step
    async def process_search_context(self, ctx: Context, ev: VectorSearchEvent) -> ContextProcessedEvent:
        """
        Process search results and prepare context for LLM.
        
        Args:
            ctx: Workflow context
            ev: Vector search event
            
        Returns:
            ContextProcessedEvent with processed context
        """
        async with TimingContext("context_processing") as timer:
            # Sort results by relevance
            sorted_results = sorted(ev.results, key=lambda x: x.get('score', 0), reverse=True)
            top_results = sorted_results[:3]
            
            # Compress context using response service
            compressed_context = self.response_service.compress_context(top_results, max_chars=800)
            
            # Format sources for response
            # We need video_metadata from the original start event - store it in context
            video_metadata = getattr(ctx.data, 'video_metadata', None)
            formatted_sources = self.response_service.format_search_sources(top_results, video_metadata)
            
            # Store timing
            ctx.data["timings"]["context_processing"] = timer.elapsed_ms
            
        return ContextProcessedEvent(
            question=ev.question,
            compressed_context=compressed_context,
            formatted_sources=formatted_sources,
            search_results=ev.results
        )
    
    @step
    async def generate_answer(self, ctx: Context, ev: ContextProcessedEvent) -> AnswerGeneratedEvent:
        """
        Generate answer using LLM with processed context.
        
        Args:
            ctx: Workflow context
            ev: Context processed event
            
        Returns:
            AnswerGeneratedEvent with generated answer
        """
        async with TimingContext("answer_generation") as timer:
            answer = "I couldn't generate an answer based on the available content."
            
            if ev.compressed_context and ev.compressed_context.strip():
                try:
                    from ai_utils.models import ChatMessage, ChatRequest
                    
                    # Prepare prompt
                    prompt = f"""Based on this video content, answer the question concisely:

{ev.compressed_context}

Question: {ev.question}
Answer:"""
                    
                    messages = [ChatMessage(role="user", content=prompt)]
                    
                    # Use Gemini only (no OpenAI fallback)
                    services = self.service_container.get_ai_services()
                    llm_service = services['llm']
                    
                    gemini_request = ChatRequest(
                        messages=messages,
                        model="gemini-2.5-flash",
                        temperature=0.3,
                        max_tokens=4000
                    )
                    
                    self.logger.info("🤖 Generating answer with Gemini...")
                    response = await llm_service.provider.chat_completion(gemini_request)
                    
                    if response and hasattr(response, 'choices') and response.choices:
                        answer = response.choices[0].message.content.strip()
                        self.logger.info(f"✅ Answer generated: {answer[:100]}...")
                    else:
                        self.logger.warning("🤖 Gemini returned no choices")
                        
                except Exception as e:
                    self.logger.error(f"🤖 Answer generation failed: {e}")
            
            # Store timing
            ctx.data["timings"]["answer_generation"] = timer.elapsed_ms
            
        return AnswerGeneratedEvent(
            question=ev.question,
            answer=answer,
            compressed_context=ev.compressed_context,
            formatted_sources=ev.formatted_sources,
            search_results=ev.search_results
        )
    
    @step
    async def finalize_response(self, ctx: Context, ev: AnswerGeneratedEvent) -> StopEvent:
        """
        Finalize and return the complete search response.
        
        Args:
            ctx: Workflow context
            ev: Answer generated event
            
        Returns:
            StopEvent with final response
        """
        async with TimingContext("response_finalization") as timer:
            # Calculate confidence
            confidence = self.response_service.calculate_confidence_score(ev.search_results)
            
            # Prepare timing information
            ctx.data["timings"]["response_finalization"] = timer.elapsed_ms
            timing_info = self.response_service.format_timing_info(
                ctx.data["timings"], 
                sum(ctx.data["timings"].values())
            )
            
            # Assemble final response
            response = {
                'answer': ev.answer,
                'sources': ev.formatted_sources[:3],
                'confidence': confidence,
                'search_method': 'llamaindex_workflow',
                'results_count': len(ev.search_results),
                'timing': timing_info
            }
            
            total_time = sum(ctx.data["timings"].values())
            self.logger.info(f"🏁 Workflow completed in {total_time:.2f}ms")
            
        return StopEvent(result=response)
    
    async def run_search(self, question: str, video_metadata, transcript) -> Dict[str, Any]:
        """
        Convenience method to run the complete search workflow.
        
        Args:
            question: User's question
            video_metadata: Video metadata instance
            transcript: Video transcript instance
            
        Returns:
            Complete search response dictionary
        """
        # Store video_metadata in a way the workflow can access it
        # This is a temporary approach - in a real implementation, we'd pass it through events
        
        result = await self.run(
            question=question,
            video_metadata=video_metadata,
            transcript=transcript
        )
        
        return result