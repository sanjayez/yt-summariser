"""
Video Summary Generation Task
Generates AI-powered summaries and key points from video transcripts using OpenAI LLM.
"""

from celery import shared_task
from django.db import transaction
import logging
import json
import asyncio  # Move import to top to avoid scope issues

from api.models import URLRequestTable
from ..models import VideoMetadata, VideoTranscript
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)
from ..text_utils.chunking import chunk_transcript_text, validate_embedding_text
from ai_utils.providers.openai_llm import OpenAILLMProvider
from ai_utils.services.llm_service import LLMService
from ai_utils.models import ChatMessage, ChatRequest
from ai_utils.config import AIConfig

logger = logging.getLogger(__name__)

def extract_key_points_from_summary(summary_text: str) -> list:
    """
    Extract key points from summary text using simple heuristics.
    Looks for bullet points, numbered lists, or sentence breaks.
    """
    try:
        key_points = []
        
        # Split by common bullet point patterns
        lines = summary_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for bullet points or numbered lists
            if (line.startswith('•') or line.startswith('-') or 
                line.startswith('*') or line.startswith('→') or
                (len(line) > 3 and line[0].isdigit() and line[1:3] in ['. ', ') ', '- '])):
                
                # Clean up the bullet point
                clean_point = line
                for prefix in ['•', '-', '*', '→']:
                    if clean_point.startswith(prefix):
                        clean_point = clean_point[1:].strip()
                        break
                
                # Remove numbered list prefixes
                if clean_point and clean_point[0].isdigit():
                    parts = clean_point.split(' ', 1)
                    if len(parts) > 1:
                        clean_point = parts[1].strip()
                
                if clean_point and len(clean_point) > 10:  # Minimum length for meaningful point
                    key_points.append(clean_point)
        
        # If no bullet points found, try to extract from sentences
        if not key_points and summary_text:
            sentences = summary_text.split('. ')
            # Take the most informative sentences (avoid very short ones)
            key_points = [s.strip() + '.' for s in sentences if len(s.strip()) > 20][:5]
        
        # Limit to reasonable number of key points
        return key_points[:8]
        
    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
        return []

def create_summary_prompt(transcript_text: str, video_metadata=None) -> str:
    """
    Create a comprehensive prompt for video summarization.
    """
    try:
        # Base prompt
        prompt = """You are an expert at summarizing video content. Please provide a comprehensive summary of the following video transcript.

Instructions:
1. Create a clear, concise summary (200-400 words)
2. Focus on the main topics, key insights, and important information
3. Use bullet points for key takeaways
4. Maintain the original context and meaning
5. Make it engaging and informative

"""
        
        # Add video context if available
        if video_metadata:
            context_parts = []
            if video_metadata.title:
                context_parts.append(f"Video Title: {video_metadata.title}")
            if video_metadata.channel_name:
                context_parts.append(f"Channel: {video_metadata.channel_name}")
            if video_metadata.duration:
                context_parts.append(f"Duration: {video_metadata.duration_string}")
            
            if context_parts:
                prompt += f"Video Context:\n{chr(10).join(context_parts)}\n\n"
        
        prompt += f"Transcript to summarize:\n{transcript_text}\n\nSummary:"
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error creating summary prompt: {e}")
        return f"Please summarize this video transcript:\n{transcript_text}"

def generate_summary_sync(transcript_text: str, video_metadata=None) -> tuple:
    """
    Synchronous helper function to generate summary using LLM service.
    Returns (summary_text, key_points).
    """
    try:
        # Initialize LLM service with OpenAI provider and config
        from ai_utils.config import get_config
        config = get_config()
        llm_provider = OpenAILLMProvider(config)
        llm_service = LLMService(llm_provider)
        
        # If transcript is very long, chunk it and summarize chunks
        if len(transcript_text) > 4000:  # Approximate token limit consideration
            logger.info(f"Transcript is long ({len(transcript_text)} chars), chunking for summarization")
            
            chunks = chunk_transcript_text(transcript_text, chunk_size=3000, chunk_overlap=200)
            chunk_summaries = []
            
            # Summarize each chunk
            for i, chunk in enumerate(chunks):
                chunk_prompt = create_summary_prompt(chunk, video_metadata)
                
                try:
                    # Create chat request
                    messages = [ChatMessage(role="user", content=chunk_prompt)]
                    chat_request = ChatRequest(messages=messages)
                    
                    # Generate summary for chunk using sync method
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        response_data = loop.run_until_complete(llm_service.chat_completion(chat_request))
                        if response_data and response_data.get('response') and response_data['response'].choices:
                            chunk_summaries.append(response_data['response'].choices[0].message.content)
                            logger.info(f"Generated summary for chunk {i+1}/{len(chunks)}")
                    finally:
                        loop.close()
                    
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk {i+1}: {e}")
                    continue
            
            # Combine chunk summaries
            if chunk_summaries:
                combined_summary = "\n\n".join(chunk_summaries)
                
                # Create final summary from combined chunks
                final_prompt = f"""Please create a cohesive summary from these section summaries of a video:

{combined_summary}

Create a unified, well-structured summary with key points."""
                
                messages = [ChatMessage(role="user", content=final_prompt)]
                chat_request = ChatRequest(messages=messages)
                
                # Use sync method for final summary
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    final_response_data = loop.run_until_complete(llm_service.chat_completion(chat_request))
                    if final_response_data and final_response_data.get('response') and final_response_data['response'].choices:
                        summary_text = final_response_data['response'].choices[0].message.content
                    else:
                        summary_text = combined_summary
                finally:
                    loop.close()
            else:
                raise ValueError("Failed to generate summaries for any chunks")
                
        else:
            # Direct summarization for shorter transcripts
            logger.info(f"Generating direct summary for transcript ({len(transcript_text)} chars)")
            
            # Create summary prompt
            prompt = create_summary_prompt(transcript_text, video_metadata)
            
            # Generate summary
            messages = [ChatMessage(role="user", content=prompt)]
            chat_request = ChatRequest(messages=messages)
            
            # Use sync method for direct summary
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response_data = loop.run_until_complete(llm_service.chat_completion(chat_request))
            finally:
                loop.close()
            
            if not response_data or not response_data.get('response') or not response_data['response'].choices:
                raise ValueError("Failed to generate summary from LLM")
            
            summary_text = response_data['response'].choices[0].message.content
        
        # Extract key points from summary
        key_points = extract_key_points_from_summary(summary_text)
        
        # Validate and clean up summary
        summary_text = validate_embedding_text(summary_text, max_length=2000)
        
        return summary_text, key_points
        
    except Exception as e:
        logger.error(f"Error in sync summary generation: {e}")
        raise

@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript']['jitter'],
             retry_kwargs=YOUTUBE_CONFIG['RETRY_CONFIG']['transcript'])
@idempotent_task
def generate_video_summary(self, transcript_result, url_request_id):
    """
    Generate AI-powered summary and key points from video transcript.
    This task runs after transcript extraction is complete.
    """
    try:
        update_task_progress(self, TASK_STATES.get('GENERATING_SUMMARY', 'Generating Summary'), 10)
        
        # Get video metadata and transcript
        url_request = URLRequestTable.objects.select_related(
            'video_metadata__video_transcript'
        ).get(id=url_request_id)
        
        if not hasattr(url_request, 'video_metadata'):
            raise ValueError("VideoMetadata not found for this request")
        
        video_metadata = url_request.video_metadata
        
        if not hasattr(video_metadata, 'video_transcript'):
            raise ValueError("VideoTranscript not found for this request")
        
        transcript = video_metadata.video_transcript
        
        if not transcript.transcript_text or not transcript.transcript_text.strip():
            raise ValueError("No transcript text available for summarization")
        
        logger.info(f"Generating summary for video {video_metadata.video_id}")
        
        update_task_progress(self, TASK_STATES.get('GENERATING_SUMMARY', 'Generating Summary'), 30)
        
        # Prepare transcript text for summarization
        transcript_text = transcript.transcript_text.strip()
        
        # Generate summary using synchronous function
        summary_text, key_points = generate_summary_sync(transcript_text, video_metadata)
        
        update_task_progress(self, TASK_STATES.get('GENERATING_SUMMARY', 'Generating Summary'), 80)
        
        # Save summary and key points to database
        with transaction.atomic():
            transcript.summary = summary_text
            transcript.key_points = key_points
            transcript.save()
            
            logger.info(f"Saved summary ({len(summary_text)} chars) and {len(key_points)} key points")
        
        update_task_progress(self, TASK_STATES.get('GENERATING_SUMMARY', 'Generating Summary'), 100)
        
        result = {
            'summary_length': len(summary_text),
            'key_points_count': len(key_points),
            'video_title': video_metadata.title or 'Unknown',
            'video_id': video_metadata.video_id
        }
        
        logger.info(f"Successfully generated summary for video {video_metadata.video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Summary generation failed for request {url_request_id}: {e}")
        
        # Mark transcript summary as failed but don't stop the chain
        try:
            with transaction.atomic():
                url_request = URLRequestTable.objects.select_related(
                    'video_metadata__video_transcript'
                ).get(id=url_request_id)
                
                if (hasattr(url_request, 'video_metadata') and 
                    hasattr(url_request.video_metadata, 'video_transcript')):
                    
                    transcript = url_request.video_metadata.video_transcript
                    transcript.summary = f"Summary generation failed: {str(e)}"
                    transcript.key_points = []
                    transcript.save()
                    
        except Exception as db_error:
            logger.error(f"Failed to update transcript with error status: {db_error}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('generate_video_summary', self.request.id, [url_request_id], {}, e)
        
        # Return error result but don't break the chain
        return {
            'summary_length': 0,
            'key_points_count': 0,
            'video_title': 'Unknown',
            'video_id': 'Unknown',
            'error': str(e)
        } 