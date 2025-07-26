from .serializers import URLRequestTableSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .utils.get_client_ip import get_client_ip
from django.http import StreamingHttpResponse
from .models import URLRequestTable
from video_processor.processors.workflow import process_youtube_video
from video_processor.validators import validate_youtube_url
import json
import time
import asyncio
import logging
from video_processor.config import API_CONFIG

logger = logging.getLogger(__name__)

@api_view(['POST'])
def process_single_video(request):
    """
    Process a single YouTube video through the complete pipeline:
    metadata extraction → transcript extraction → summary generation → content embedding
    """
    # Get client IP address
    ip_address = get_client_ip(request)
    
    # Get URL from request body
    url = request.data.get('url')

    if not url:
        return Response({'error': 'URL is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate YouTube URL format
    try:
        validate_youtube_url(url)
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    # Prepare data for serializer
    data = {
        'url': url,
        'ip_address': ip_address,
        'status': 'processing'
    }
    
    # Use serializer to save data
    serializer = URLRequestTableSerializer(data=data)
    if serializer.is_valid():
        url_request = serializer.save()
        
        # Trigger enhanced video processing pipeline
        process_youtube_video.delay(url_request.id)
        
        # Return response in requested format with request_id
        return Response({
            'request_id': str(url_request.request_id),
            'url': url,
            'status': 'processing',
            'message': 'Video processing started. Use the request_id to check status and retrieve results.'
        }, status=status.HTTP_201_CREATED)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_video_summary(request, request_id):
    """
    Get the AI-generated summary and key points for a processed video.
    Returns summary, key points, and video metadata.
    """
    try:
        # Get the request and related data
        url_request = URLRequestTable.objects.select_related(
            'video_metadata__video_transcript'
        ).get(request_id=request_id)
        
        # Check if video metadata exists
        if not hasattr(url_request, 'video_metadata'):
            return Response({
                'error': 'Video metadata not found',
                'status': 'not_found',
                'message': 'Video processing may not have started or failed during metadata extraction'
            }, status=status.HTTP_404_NOT_FOUND)
        
        video_metadata = url_request.video_metadata
        
        # Check if transcript exists
        if not hasattr(video_metadata, 'video_transcript'):
            return Response({
                'error': 'Video transcript not found',
                'status': 'not_found',
                'message': 'Video processing may not have reached transcript extraction stage'
            }, status=status.HTTP_404_NOT_FOUND)
        
        transcript = video_metadata.video_transcript
        
        # Check processing status
        if url_request.status == 'processing':
            return Response({
                'status': 'processing',
                'message': 'Video is still being processed. Please check back later.',
                'stages': {
                    'metadata_extracted': video_metadata.status == 'success',
                    'transcript_extracted': transcript.status == 'success',
                    'summary_generated': bool(transcript.summary and transcript.summary.strip())
                }
            }, status=status.HTTP_202_ACCEPTED)
        
        elif url_request.status == 'failed':
            return Response({
                'error': 'Video processing failed',
                'status': 'failed',
                'message': 'Video processing encountered errors and could not be completed'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Check if summary is available
        if not transcript.summary or not transcript.summary.strip():
            return Response({
                'error': 'Summary not available',
                'status': 'no_summary',
                'message': 'Video was processed but summary generation failed or is not yet complete'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Prepare video metadata for response
        video_metadata_response = {
            'video_id': video_metadata.video_id,
            'title': video_metadata.title,
            'description': video_metadata.description[:200] + '...' if video_metadata.description and len(video_metadata.description) > 200 else video_metadata.description,
            'duration': video_metadata.duration,
            'duration_string': video_metadata.duration_string,
            'channel_name': video_metadata.channel_name,
            'view_count': video_metadata.view_count,
            'like_count': video_metadata.like_count,
            'upload_date': video_metadata.upload_date,
            'language': video_metadata.language,
            'tags': video_metadata.tags[:10] if video_metadata.tags else [],  # Limit tags
            'youtube_url': video_metadata.webpage_url,
            'thumbnail': video_metadata.thumbnail
        }
        
        # Return successful response
        response_data = {
            'summary': transcript.summary,
            'key_points': transcript.key_points if transcript.key_points else [],
            'video_metadata': video_metadata_response,
            'status': 'completed',
            'generated_at': transcript.created_at.isoformat() if transcript.created_at else None,
            'summary_length': len(transcript.summary) if transcript.summary else 0,
            'key_points_count': len(transcript.key_points) if transcript.key_points else 0
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except URLRequestTable.DoesNotExist:
        return Response({
            'error': 'Request not found',
            'status': 'not_found',
            'message': f'No video processing request found with ID: {request_id}'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        return Response({
            'error': 'Internal server error',
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def search_video_content_async(question: str, video_metadata, transcript) -> dict:
    """
    Smart search strategy: start with metadata/summary, then search segments if needed.
    Returns search results with sources and confidence.
    """
    try:
        from ai_utils.services.vector_service import VectorService
        from ai_utils.services.embedding_service import EmbeddingService
        from ai_utils.providers.openai_llm import OpenAILLMProvider
        from ai_utils.providers.pinecone_store import PineconeVectorStoreProvider
        from ai_utils.providers.openai_embeddings import OpenAIEmbeddingProvider
        from ai_utils.services.llm_service import LLMService
        from ai_utils.models import VectorQuery, ChatMessage, ChatRequest
        from ai_utils.config import get_config
        
        config = get_config()
        embedding_provider = OpenAIEmbeddingProvider(config)
        embedding_service = EmbeddingService(provider=embedding_provider)
        vector_provider = PineconeVectorStoreProvider(config)
        vector_service = VectorService(provider=vector_provider)
        llm_provider = OpenAILLMProvider(config=config)
        llm_service = LLMService(provider=llm_provider)
        video_id = video_metadata.video_id
        
        all_sources = []
        search_results = []
        
        # Search video segments (the main content that's embedded)
        try:
            logger.info(f"🔍 Starting vector search for video {video_id} with question: '{question}'")
            
            segments_results = await vector_service.search_by_text(
                text=question,
                embedding_service=embedding_service,
                top_k=5,
                filters={'video_id': video_id, 'type': 'segment'}
            )
            
            logger.info(f"🔍 Vector search completed. Results: {len(segments_results.results) if segments_results else 0}")
            
            if segments_results and segments_results.results:
                logger.info(f"🔍 Processing {len(segments_results.results)} results")
                for i, result in enumerate(segments_results.results):
                    logger.info(f"🔍 Result {i+1}: score={result.score:.4f}, text='{result.text[:50]}...'")
                    search_results.append({
                        'type': 'segment',
                        'text': result.text,
                        'score': result.score,
                        'metadata': result.metadata
                    })
            else:
                logger.warning(f"🔍 No vector search results found for video {video_id}")
                    
        except Exception as e:
            logger.error(f"🔍 Segments search failed: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        
        # Sort by relevance score and take top 3
        search_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = search_results[:3]
        
        # Prepare sources for LLM context
        context_parts = []
        for result in top_results:
            if result['type'] == 'metadata':
                context_parts.append(f"Video Info: {result['text']}")
            elif result['type'] == 'summary':
                context_parts.append(f"Summary: {result['text']}")
            elif result['type'] == 'segment':
                timestamp = result['metadata'].get('start_time', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                context_parts.append(f"[{time_str}] {result['text']}")
                
                # Add to sources for response
                all_sources.append({
                    'type': 'segment',
                    'timestamp': time_str,
                    'text': result['text'].replace(f"[{time_str}]", "").strip(),
                    'youtube_url': result['metadata'].get('youtube_url', ''),
                    'confidence': result['score']
                })
        
        # Generate answer using LLM
        if context_parts:
            context = "\n\n".join(context_parts)
            
            prompt = f"""Based on the following video content, please answer the user's question accurately and concisely.

Video Content:
{context}

Question: {question}

Please provide a helpful answer based on the video content. If the information isn't sufficient to answer the question, say so."""
            
            messages = [ChatMessage(role="user", content=prompt)]
            chat_request = ChatRequest(messages=messages)
            
            response_data = await llm_service.chat_completion(chat_request)
            if response_data and response_data.get('response') and response_data['response'].choices:
                answer = response_data['response'].choices[0].message.content
            else:
                answer = "I couldn't generate an answer based on the available content."
            
            # Calculate overall confidence
            avg_confidence = sum(r['score'] for r in top_results) / len(top_results) if top_results else 0.0
            
            return {
                'answer': answer,
                'sources': all_sources[:3],  # Top 3 sources
                'confidence': round(avg_confidence, 2),
                'search_method': 'vector_search',
                'results_count': len(top_results)
            }
        
        else:
            return {
                'answer': "I couldn't find relevant information in the video to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'search_method': 'vector_search',
                'results_count': 0
            }
            
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise

def search_transcript_fallback(question: str, transcript_text: str, video_metadata) -> dict:
    """
    Fallback search using raw transcript text when embeddings are not available.
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
        logger.error(f"Transcript fallback search failed: {e}")
        return {
            'answer': f"Error searching transcript: {str(e)}",
            'sources': [],
            'confidence': 0.0,
            'search_method': 'transcript_fallback',
            'results_count': 0
        }

@api_view(['POST'])
def ask_video_question(request, request_id):
    """
    Ask questions about a processed video using smart search strategy.
    Uses vector search when available, falls back to transcript search.
    """
    try:
        # Get question from request
        question = request.data.get('question', '').strip()
        if not question:
            return Response({
                'error': 'Question is required',
                'message': 'Please provide a question in the request body'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get the request and related data
        url_request = URLRequestTable.objects.select_related(
            'video_metadata__video_transcript'
        ).get(request_id=request_id)
        
        # Check if video metadata exists
        if not hasattr(url_request, 'video_metadata'):
            return Response({
                'error': 'Video metadata not found',
                'status': 'not_found',
                'message': 'Video processing may not have started or failed during metadata extraction'
            }, status=status.HTTP_404_NOT_FOUND)
        
        video_metadata = url_request.video_metadata
        
        # Check if transcript exists
        if not hasattr(video_metadata, 'video_transcript'):
            return Response({
                'error': 'Video transcript not found',
                'status': 'not_found',
                'message': 'Video processing may not have reached transcript extraction stage'
            }, status=status.HTTP_404_NOT_FOUND)
        
        transcript = video_metadata.video_transcript
        
        # Check if video processing is complete
        if url_request.status == 'processing':
            return Response({
                'error': 'Video still processing',
                'status': 'processing',
                'message': 'Please wait for video processing to complete before asking questions'
            }, status=status.HTTP_202_ACCEPTED)
        
        elif url_request.status == 'failed':
            return Response({
                'error': 'Video processing failed',
                'status': 'failed',
                'message': 'Cannot answer questions for failed video processing'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Check if transcript is available
        if not transcript.transcript_text or not transcript.transcript_text.strip():
            return Response({
                'error': 'No transcript available',
                'status': 'no_transcript',
                'message': 'Cannot answer questions without video transcript'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Try vector search first (if embeddings are available)
        search_result = None
        
        if video_metadata.is_embedded:
            try:
                logger.info(f"Using vector search for question: {question}")
                search_result = asyncio.run(
                    search_video_content_async(question, video_metadata, transcript)
                )
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to transcript search: {e}")
                search_result = None
        
        # Fallback to transcript search
        if not search_result:
            logger.info(f"Using transcript fallback for question: {question}")
            search_result = search_transcript_fallback(
                question, transcript.transcript_text, video_metadata
            )
        
        # Prepare response
        response_data = {
            'question': question,
            'answer': search_result['answer'],
            'sources': search_result['sources'],
            'confidence': search_result['confidence'],
            'search_method': search_result['search_method'],
            'results_count': search_result['results_count'],
            'video_metadata': {
                'video_id': video_metadata.video_id,
                'title': video_metadata.title,
                'duration_string': video_metadata.duration_string,
                'youtube_url': video_metadata.webpage_url
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except URLRequestTable.DoesNotExist:
        return Response({
            'error': 'Request not found',
            'status': 'not_found',
            'message': f'No video processing request found with ID: {request_id}'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        logger.error(f"Error in ask_video_question: {e}")
        return Response({
            'error': 'Internal server error',
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Stream the status of the request
def video_status_stream(request, request_id):
    """
    Stream real-time status updates for video processing including all stages:
    metadata extraction, transcript extraction, summary generation, content embedding
    """
    def event_stream():
        max_attempts = API_CONFIG['POLLING']['status_check_max_attempts']
        poll_interval = API_CONFIG['POLLING']['status_check_interval']
        attempts = 0
        
        # Maximum time based on config (max_attempts * poll_interval seconds)
        while attempts < max_attempts:
            try:
                url_request = URLRequestTable.objects.select_related(
                    'video_metadata',
                    'video_metadata__video_transcript'
                ).get(request_id=request_id)
                
                # Build enhanced status data
                data = {
                    'overall_status': url_request.status,
                    'timestamp': time.time(),
                    'metadata_status': None,
                    'transcript_status': None,
                    'summary_status': None,
                    'embedding_status': None,
                    'stages': {
                        'metadata_extracted': False,
                        'transcript_extracted': False,
                        'summary_generated': False,
                        'content_embedded': False,
                        'processing_complete': False
                    }
                }
                
                # Add metadata details if exists
                if hasattr(url_request, 'video_metadata'):
                    metadata = url_request.video_metadata
                    data['metadata_status'] = metadata.status
                    data['stages']['metadata_extracted'] = metadata.status == 'success'
                    
                    # Add embedding status
                    if hasattr(metadata, 'is_embedded'):
                        data['embedding_status'] = 'success' if metadata.is_embedded else 'pending'
                        data['stages']['content_embedded'] = metadata.is_embedded
                
                # Add transcript details if exists through VideoMetadata
                if hasattr(url_request, 'video_metadata') and hasattr(url_request.video_metadata, 'video_transcript'):
                    transcript = url_request.video_metadata.video_transcript
                    data['transcript_status'] = transcript.status
                    data['stages']['transcript_extracted'] = transcript.status == 'success'
                    
                    # Add summary status
                    if transcript.summary:
                        data['summary_status'] = 'success'
                        data['stages']['summary_generated'] = True
                    else:
                        data['summary_status'] = 'pending' if transcript.status == 'success' else 'waiting'
                
                # Overall completion status
                data['stages']['processing_complete'] = url_request.status in ['success', 'failed']
                
                # Add progress percentage
                completed_stages = sum(1 for stage in data['stages'].values() if stage)
                total_stages = len(data['stages'])
                data['progress_percentage'] = int((completed_stages / total_stages) * 100)
                
                # Send data
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop streaming if complete
                if url_request.status in ['success', 'failed']:
                    break
                    
                time.sleep(poll_interval)  # Wait based on config
                attempts += 1
                
            except URLRequestTable.DoesNotExist:
                yield f"data: {json.dumps({'error': 'Request not found'})}\n\n"
                break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response