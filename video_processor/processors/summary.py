"""
Video Summary Generation Task
Generates AI-powered summaries and key points from video transcripts using Gemini LLM.
"""

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from celery_progress.backend import ProgressRecorder
from django.db import transaction
import logging
import os
import json
import re
import asyncio  # Move import to top to avoid scope issues
import time

from api.models import URLRequestTable
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..utils import (
    idempotent_task, handle_dead_letter_task, 
    update_task_progress
)
from ..text_utils.chunking import chunk_transcript_text, validate_embedding_text
from ai_utils.services.registry import get_gemini_llm_service
from ai_utils.models import ChatMessage, ChatRequest
from temp_file_logger import append_line

logger = logging.getLogger(__name__)

def extract_key_points_from_summary(structured_response: str) -> list:
    """
    Extract structured chapters from LLM response with robust fallback parsing.
    Tries multiple parsing strategies to extract useful content.
    
    TODO: Simplify this to single parsing strategy once LLM responses are more consistent
    """
    try:
        # Strategy 1: Look for CHAPTERS: section
        if "CHAPTERS:" in structured_response:
            chapters_section = structured_response.split("CHAPTERS:")[1].strip()
            
            # Try to parse as JSON
            try:
                chapters_json = json.loads(chapters_section)
                if isinstance(chapters_json, list) and len(chapters_json) > 0:
                    logger.info(f"Successfully parsed {len(chapters_json)} chapters from CHAPTERS: section")
                    return chapters_json
            except json.JSONDecodeError:
                logger.warning("Failed to parse CHAPTERS section as JSON, trying cleanup")
                # Try to clean up the JSON and parse again
                cleaned_json = _cleanup_json_string(chapters_section)
                try:
                    chapters_json = json.loads(cleaned_json)
                    if isinstance(chapters_json, list) and len(chapters_json) > 0:
                        logger.info(f"Successfully parsed {len(chapters_json)} chapters after JSON cleanup")
                        return chapters_json
                except json.JSONDecodeError:
                    logger.warning("JSON cleanup failed, trying text extraction")
        
        # Strategy 2: Look for any JSON array in the response
        json_arrays = _extract_json_arrays(structured_response)
        for json_array in json_arrays:
            if len(json_array) > 0 and isinstance(json_array[0], dict):
                # Check if it looks like chapters (has title, summary, etc.)
                if any(key in json_array[0] for key in ['title', 'summary', 'chapter']):
                    logger.info(f"Found {len(json_array)} chapters via JSON array extraction")
                    return json_array
        
        # Strategy 3: Extract chapters from numbered sections in text
        text_chapters = _extract_chapters_from_text(structured_response)
        if text_chapters:
            logger.info(f"Extracted {len(text_chapters)} chapters from text structure")
            return text_chapters
        
        # Strategy 4: Create single chapter from entire response if it contains useful content
        if len(structured_response.strip()) > 50:  # Has substantial content
            fallback_chapter = {
                "chapter": 1,
                "title": "Video Summary",
                "summary": structured_response.strip()[:500] + ("..." if len(structured_response) > 500 else "")
            }
            logger.info("Created fallback single chapter from response content")
            return [fallback_chapter]
        
        # Final fallback: return empty array
        logger.warning("Could not extract any chapters from response")
        return []
        
    except Exception as e:
        logger.error(f"Error parsing structured response: {e}")
        return []

def _cleanup_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues from LLM responses."""
    # Remove common prefixes/suffixes
    json_str = json_str.strip()
    
    # Remove markdown code blocks
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    if json_str.startswith('```'):
        json_str = json_str[3:]
    if json_str.endswith('```'):
        json_str = json_str[:-3]
    
    # Find the first [ and last ]
    start_idx = json_str.find('[')
    end_idx = json_str.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = json_str[start_idx:end_idx+1]
    
    return json_str.strip()

def _extract_json_arrays(text: str) -> list:
    """Extract all JSON arrays from text, even if embedded in other content."""
    arrays = []
    
    # Find all potential JSON arrays
    start_indices = [i for i, char in enumerate(text) if char == '[']
    
    for start_idx in start_indices:
        # Find matching closing bracket
        bracket_count = 0
        for end_idx in range(start_idx, len(text)):
            if text[end_idx] == '[':
                bracket_count += 1
            elif text[end_idx] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    # Found complete array
                    try:
                        json_str = text[start_idx:end_idx+1]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, list):
                            arrays.append(parsed)
                    except json.JSONDecodeError:
                        continue
                    break
    
    return arrays

def _extract_chapters_from_text(text: str) -> list:
    """Extract chapters from text using common numbering patterns."""
    chapters = []
    
    # Look for numbered sections (1., 2., Chapter 1, etc.)
    
    # Pattern for numbered sections
    patterns = [
        r'(?:^|\n)\s*(\d+)\.?\s*([^\n]+)\n(.*?)(?=(?:\n\s*\d+\.|\Z))',
        r'(?:^|\n)\s*(?:Chapter|Section)\s+(\d+):?\s*([^\n]+)\n(.*?)(?=(?:\n\s*(?:Chapter|Section)\s+\d+|\Z))',
        r'(?:^|\n)\s*##\s*(\d+)\.?\s*([^\n]+)\n(.*?)(?=(?:\n\s*##\s*\d+|\Z))'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            for i, (num, title, content) in enumerate(matches):
                chapters.append({
                    "chapter": int(num) if num.isdigit() else i + 1,
                    "title": title.strip(),
                    "summary": content.strip()[:300] + ("..." if len(content.strip()) > 300 else "")
                })
            break  # Use first successful pattern
    
    return chapters

def _parse_llm_response_robust(structured_response: str) -> tuple:
    """
    Robust parsing of LLM response with multiple fallback strategies.
    Returns (executive_summary, chapters) tuple.
    
    TODO: Replace with simple JSON parsing once LLM prompt is improved
    """
    try:
        logger.info(f"Parsing LLM response ({len(structured_response)} chars)")
        
        # Strategy 1: Perfect structured format (EXECUTIVE SUMMARY: + CHAPTERS:)
        if "EXECUTIVE SUMMARY:" in structured_response and "CHAPTERS:" in structured_response:
            try:
                parts = structured_response.split("CHAPTERS:")
                executive_summary = parts[0].replace("EXECUTIVE SUMMARY:", "").strip()
                
                # Extract chapters using the robust function
                chapters = extract_key_points_from_summary(structured_response)
                
                if executive_summary and len(executive_summary.strip()) > 10:
                    logger.info("Successfully parsed structured response (perfect format)")
                    return executive_summary, chapters
                    
            except Exception as e:
                logger.warning(f"Perfect format parsing failed: {e}, trying fallbacks")
        
        # Strategy 2: Look for EXECUTIVE SUMMARY section only
        if "EXECUTIVE SUMMARY:" in structured_response:
            try:
                # Extract everything after "EXECUTIVE SUMMARY:"
                summary_part = structured_response.split("EXECUTIVE SUMMARY:")[1]
                
                # If there's a CHAPTERS section, stop at it
                if "CHAPTERS:" in summary_part:
                    executive_summary = summary_part.split("CHAPTERS:")[0].strip()
                else:
                    executive_summary = summary_part.strip()
                
                # Extract chapters (will use fallback methods if needed)
                chapters = extract_key_points_from_summary(structured_response)
                
                if executive_summary and len(executive_summary.strip()) > 10:
                    logger.info("Successfully parsed executive summary section")
                    return executive_summary, chapters
                    
            except Exception as e:
                logger.warning(f"Executive summary parsing failed: {e}")
        
        # Strategy 3: Look for chapters first, use rest as summary
        chapters = extract_key_points_from_summary(structured_response)
        if chapters:
            # Remove the chapters section from response to get summary
            executive_summary = structured_response
            
            # Remove CHAPTERS: section if present
            if "CHAPTERS:" in executive_summary:
                executive_summary = executive_summary.split("CHAPTERS:")[0]
            
            # Remove EXECUTIVE SUMMARY: prefix if present
            if "EXECUTIVE SUMMARY:" in executive_summary:
                executive_summary = executive_summary.replace("EXECUTIVE SUMMARY:", "")
            
            executive_summary = executive_summary.strip()
            
            # If summary is too short, use first chapter summary or response beginning
            if len(executive_summary.strip()) < 20:
                if chapters and chapters[0].get('summary'):
                    executive_summary = chapters[0]['summary'][:200] + "..."
                else:
                    executive_summary = structured_response[:200] + "..."
            
            logger.info(f"Extracted summary using chapters-first strategy")
            return executive_summary, chapters
        
        # Strategy 4: Use entire response as summary, create basic chapters
        if len(structured_response.strip()) > 50:
            # Use first part as executive summary
            lines = structured_response.strip().split('\n')
            
            # Try to find a good summary (first substantial paragraph)
            executive_summary = ""
            for line in lines:
                if len(line.strip()) > 30:  # Substantial line
                    executive_summary = line.strip()
                    break
            
            # If no good line found, use first 200 chars
            if not executive_summary:
                executive_summary = structured_response.strip()[:200] + "..."
            
            # Create single chapter from full response
            chapters = [{
                "chapter": 1,
                "title": "Video Summary",
                "summary": structured_response.strip()
            }]
            
            logger.info("Used entire response as summary with single chapter")
            return executive_summary, chapters
        
        # Strategy 5: Minimal fallback for very short responses
        if len(structured_response.strip()) > 10:
            logger.warning("Using minimal fallback for short response")
            return structured_response.strip(), []
        
        # Final fallback: Empty response handling
        logger.error("Response too short or empty, using error message")
        return "Summary generation produced insufficient content", []
        
    except Exception as e:
        logger.error(f"All parsing strategies failed: {e}")
        # Last resort: return whatever we can
        return structured_response[:200] if structured_response else "Parsing failed", []

def create_summary_prompt(transcript_text: str, video_metadata=None) -> str:
    """
    Create a comprehensive prompt for video summarization.
    """
    try:
        # Base prompt
        prompt = """You are an expert at analyzing video content. Please provide:

1. EXECUTIVE SUMMARY (2-3 sentences): High-level overview of the video's main theme

2. CHAPTERS (JSON array): Break the content into logical chapters

Format your response EXACTLY like this:

EXECUTIVE SUMMARY:
[Your 2-3 sentence summary here]

CHAPTERS:
[
  {
    "chapter": 1,
    "title": "Chapter Name",
    "summary": "Detailed explanation of this chapter's content..."
  },
  {
    "chapter": 2,
    "title": "Another Chapter",
    "summary": "Another detailed explanation..."
  }
]

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
        llm_service = get_gemini_llm_service()
        
        # TODO: Simplify chunking logic - current complexity handles LLM API instability
        # Enhanced chunking for very long transcripts (like 26+ minute videos)
        if len(transcript_text) > 15000:  # More aggressive chunking for stability
            logger.info(f"Long transcript detected ({len(transcript_text)} chars), using enhanced chunking")
            
            # Use smaller chunks for better stability
            chunks = chunk_transcript_text(transcript_text, chunk_size=2000, chunk_overlap=100)
            chunk_summaries = []
            
            # Process chunks with progress tracking and error recovery
            total_chunks = len(chunks)
            logger.info(f"Processing {total_chunks} chunks for long video")
            
            for i, chunk in enumerate(chunks):
                chunk_prompt = create_summary_prompt(chunk, video_metadata)
                
                try:
                    logger.info(f"Processing chunk {i+1}/{total_chunks} ({len(chunk)} chars)")
                    
                    # Create chat request
                    messages = [ChatMessage(role="user", content=chunk_prompt)]
                    chat_request = ChatRequest(messages=messages)
                    
                    # Generate summary for chunk using sync method with timeout protection
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        # Add timeout protection for each chunk
                        start_time = time.time()
                        response_data = loop.run_until_complete(
                            asyncio.wait_for(
                                llm_service.chat_completion(chat_request), 
                                timeout=300  # 5 minute timeout per chunk
                            )
                        )
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Chunk {i+1} processed in {processing_time:.1f}s")
                        
                        if response_data and response_data.get('response') and response_data['response'].choices:
                            chunk_summaries.append(response_data['response'].choices[0].message.content)
                            logger.info(f"✅ Generated summary for chunk {i+1}/{total_chunks}")
                        else:
                            logger.warning(f"⚠️ Empty response for chunk {i+1}, skipping")
                            
                    finally:
                        loop.close()
                    
                    # Add small delay between chunks to prevent overwhelming the API
                    if i < total_chunks - 1:  # Don't sleep after last chunk
                        time.sleep(1)
                    
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout processing chunk {i+1}, skipping")
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Failed to summarize chunk {i+1}: {e}")
                    # For long videos, we can afford to skip some chunks
                    if len(chunks) > 10:  # Only skip if we have many chunks
                        continue
                    else:
                        raise  # Re-raise for shorter videos
            
            # Combine chunk summaries with better error handling
            if chunk_summaries:
                combined_summary = "\n\n".join(chunk_summaries)
                logger.info(f"Combined {len(chunk_summaries)}/{total_chunks} chunk summaries")
                
                # Create final summary from combined chunks
                final_prompt = f"""Please create a cohesive summary from these section summaries of a video:

{combined_summary}

Create a unified, well-structured summary with key points. Focus on the main themes and insights."""
                
                messages = [ChatMessage(role="user", content=final_prompt)]
                chat_request = ChatRequest(messages=messages)
                
                # Use sync method for final summary with timeout
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    final_response_data = loop.run_until_complete(
                        asyncio.wait_for(
                            llm_service.chat_completion(chat_request),
                            timeout=180  # 3 minute timeout for final summary
                        )
                    )
                    if final_response_data and final_response_data.get('response') and final_response_data['response'].choices:
                        structured_response = final_response_data['response'].choices[0].message.content
                        
                        # Try to parse structured response for chunks too
                        if "EXECUTIVE SUMMARY:" in structured_response and "CHAPTERS:" in structured_response:
                            parts = structured_response.split("CHAPTERS:")
                            summary_text = parts[0].replace("EXECUTIVE SUMMARY:", "").strip()
                        else:
                            summary_text = structured_response
                    else:
                        summary_text = combined_summary
                finally:
                    loop.close()
            else:
                # Fallback: create a basic summary if all chunks failed
                logger.warning("All chunks failed, creating fallback summary")
                summary_text = f"Summary of {video_metadata.title if video_metadata else 'video'}: Content processing encountered issues. Video contains {len(transcript_text)} characters of transcript data."
                
            # For chunked processing, use robust parsing for any response format
            executive_summary, chapters = _parse_llm_response_robust(summary_text)
            
            # Validate and clean up executive summary
            executive_summary = validate_embedding_text(executive_summary, max_length=2000)
            
            logger.info(f"Chunked summary parsing completed: exec summary {len(executive_summary)} chars, {len(chapters)} chapters")
            
            return executive_summary, chapters
                
        else:
            # Direct summarization for shorter transcripts
            logger.info(f"Generating direct summary for transcript ({len(transcript_text)} chars)")
            try:
                append_line(
                    file_path=os.getenv('TIMING_LOG_FILE', 'logs/stage_timings.log'),
                    message=f"summary_input video_id={getattr(video_metadata, 'video_id', 'unknown')} transcript_chars={len(transcript_text)}"
                )
            except Exception:
                pass
            
            # Create summary prompt
            prompt = create_summary_prompt(transcript_text, video_metadata)
            
            # Generate summary
            messages = [ChatMessage(role="user", content=prompt)]
            chat_request = ChatRequest(messages=messages)
            
            # Use sync method for direct summary
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response_data = loop.run_until_complete(
                    asyncio.wait_for(
                        llm_service.chat_completion(chat_request),
                        timeout=300  # 5 minute timeout
                    )
                )
            finally:
                loop.close()
            
            if not response_data or not response_data.get('response') or not response_data['response'].choices:
                try:
                    append_line(
                        file_path=os.getenv('TIMING_LOG_FILE', 'logs/stage_timings.log'),
                        message=f"summary_error video_id={getattr(video_metadata, 'video_id', 'unknown')} reason=no_choices"
                    )
                except Exception:
                    pass
                raise ValueError("Failed to generate summary from LLM")
            
            structured_response = response_data['response'].choices[0].message.content
            try:
                preview = (structured_response[:200] + '...') if len(structured_response) > 200 else structured_response
                append_line(
                    file_path=os.getenv('TIMING_LOG_FILE', 'logs/stage_timings.log'),
                    message=f"summary_output video_id={getattr(video_metadata, 'video_id', 'unknown')} preview={preview}"
                )
            except Exception:
                pass
        
        # Parse executive summary and chapters with robust fallback logic
        executive_summary, chapters = _parse_llm_response_robust(structured_response)
        
        # Validate and clean up executive summary
        executive_summary = validate_embedding_text(executive_summary, max_length=2000)
        
        logger.info(f"Summary parsing completed: exec summary {len(executive_summary)} chars, {len(chapters)} chapters")
        
        return executive_summary, chapters
        
    except Exception as e:
        logger.error(f"Error in sync summary generation: {e}")
        raise

@shared_task(bind=True,
             name='video_processor.generate_video_summary',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['summary_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['summary_hard_limit'],
             max_retries=YOUTUBE_CONFIG['RETRY_CONFIG']['summary']['max_retries'],
             default_retry_delay=YOUTUBE_CONFIG['RETRY_CONFIG']['summary']['countdown'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['summary']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['summary']['jitter'])
@idempotent_task
def generate_video_summary(self, transcript_result, url_request_id):
    """
    Generate AI-powered summary and key points from video transcript.
    
    Args:
        transcript_result (dict): Result from previous transcript extraction task
        url_request_id (str): UUID of the URLRequestTable to process
        
    Returns:
        dict: Summary generation results with length and key points count
        
    Raises:
        Exception: If summary generation fails after retries
    """
    url_request = None
    video_metadata = None
    transcript = None
    
    # Check if video was excluded in previous stage
    if transcript_result and transcript_result.get('excluded'):
        logger.info(f"Video was excluded in previous stage: {transcript_result.get('exclusion_reason')}")
        # Return immediately without processing summary
        return {
            'video_id': transcript_result.get('video_id'),
            'excluded': True,
            'exclusion_reason': transcript_result.get('exclusion_reason'),
            'skip_reason': 'excluded_in_previous_stage'
        }
    
    try:
        # Set initial progress
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(0, 100, "Generating summary")
        
        update_task_progress(self, TASK_STATES.get('GENERATING_SUMMARY', 'Generating Summary'), 10)
        
        # Get video metadata and transcript
        url_request = URLRequestTable.objects.select_related(
            'video_metadata'
        ).get(request_id=url_request_id)
        
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
        
        # Set final progress
        progress_recorder.set_progress(100, 100, "Summary complete")
        
        result = {
            'summary_length': len(summary_text),
            'key_points_count': len(key_points),
            'video_title': video_metadata.title or 'Unknown',
            'video_id': video_metadata.video_id
        }
        
        logger.info(f"Successfully generated summary for video {video_metadata.video_id}")
        return result
        
    except SoftTimeLimitExceeded:
        # Task is approaching timeout - save status and exit gracefully
        logger.warning(f"Summary generation soft timeout reached for request {url_request_id}")
        
        try:
            # Mark summary as failed due to timeout
            if transcript:
                transcript.summary = "Summary generation timed out - transcript may be too long or LLM API unavailable"
                transcript.key_points = []
                transcript.save()
                logger.error(f"Marked summary generation as failed due to timeout: {video_metadata.video_id if video_metadata else 'unknown'}")
                
        except Exception as cleanup_error:
            logger.error(f"Failed to update summary status during timeout cleanup: {cleanup_error}")
        
        # Re-raise with specific timeout message
        raise Exception(f"Summary generation timeout for request {url_request_id}")
        
    except Exception as e:
        logger.error(f"Summary generation failed for request {url_request_id}: {e}")
        
                    # TODO: Consider cleaner error handling once LLM reliability improves
            # Mark transcript summary as failed but don't stop the chain
        try:
            with transaction.atomic():
                if not url_request:
                    url_request = URLRequestTable.objects.select_related(
                        'video_metadata'
                    ).get(request_id=url_request_id)
                
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