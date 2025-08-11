"""
LLM-Based Video Content Classifier

This module provides intelligent video content classification using Large Language Models
to determine if videos should be excluded from the processing pipeline based on business logic.

The classifier analyzes comprehensive video context (title, channel, description, tags, transcript)
and makes informed decisions about video suitability for the Q&A system.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from celery import shared_task
from django.conf import settings

from ai_utils.config import get_config
from ai_utils.services.registry import get_gemini_llm_service
from ai_utils.models import ChatMessage, ChatRole, ChatRequest

from api.models import URLRequestTable
from ..models import VideoMetadata, VideoTranscript
from ..utils.video_filtering import add_video_to_exclusion_table
from ..utils.language_detection import should_exclude_for_language

logger = logging.getLogger(__name__)


def prepare_classification_context(video_metadata: VideoMetadata, transcript: VideoTranscript) -> Dict[str, Any]:
    """
    Prepare comprehensive context for LLM classification with validation.
    
    Args:
        video_metadata: VideoMetadata instance with video information
        transcript: VideoTranscript instance with transcript data
        
    Returns:
        Dict containing all relevant context for classification
        
    Raises:
        ValueError: If required video data is missing
    """
    # Validate inputs
    if not video_metadata:
        raise ValueError("VideoMetadata is required for classification")
    if not transcript:
        raise ValueError("VideoTranscript is required for classification")
    
    # Extract and validate transcript text
    transcript_text = transcript.transcript_text or ''
    if not transcript_text.strip():
        logger.warning(f"Empty transcript for video {video_metadata.video_id}")
    
    # Extract duration with fallback
    duration = video_metadata.duration or 0
    if duration <= 0:
        logger.warning(f"Invalid or missing duration for video {video_metadata.video_id}")
    
    # Build comprehensive context (using normalized metadata - no need for fallbacks)
    context = {
        'video_id': video_metadata.video_id,
        'title': video_metadata.title,
        'channel_name': video_metadata.channel_name,
        'description': video_metadata.description[:300],  # Truncate for token efficiency
        'duration_seconds': duration,
        'tags': video_metadata.tags[:15],  # Limit tags for efficiency
        'categories': video_metadata.categories,
        'language': video_metadata.language,
        'transcript_sample': transcript_text[:500],  # First 500 chars for analysis
        'transcript_length': len(transcript_text),
        'view_count': video_metadata.view_count,
        'like_count': video_metadata.like_count,
    }
    
    # Calculate transcript metrics
    if duration > 0:
        context['transcript_density'] = len(transcript_text) / duration
        words = transcript_text.split() if transcript_text else []
        context['estimated_words'] = len(words)
        context['words_per_minute'] = (len(words) * 60) / duration if words else 0
    else:
        context['transcript_density'] = 0
        context['estimated_words'] = 0
        context['words_per_minute'] = 0
    
    # Calculate engagement metrics if available
    if context['view_count'] > 0 and context['like_count'] > 0:
        context['like_ratio'] = context['like_count'] / context['view_count']
    else:
        context['like_ratio'] = 0
    
    logger.debug(f"Prepared classification context for video {video_metadata.video_id}: "
                f"duration={duration}s, transcript_length={len(transcript_text)}, "
                f"density={context['transcript_density']:.1f} chars/sec")
    
    return context


def build_classification_prompt(context: Dict[str, Any]) -> str:
    """
    Build comprehensive LLM prompt for video classification.
    
    Args:
        context: Video context prepared by prepare_classification_context()
        
    Returns:
        Structured prompt for LLM classification
    """
    # Load prompt template
    template_path = os.path.join(
        settings.BASE_DIR, 
        'video_processor', 
        'templates', 
        'prompts', 
        'video_classification.txt'
    )
    
    try:
        with open(template_path, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt template not found at {template_path}")
        # Fallback to a minimal prompt if template is missing
        return f"""Classify this video as MUSIC_VIDEO, BACKGROUND_ONLY, or BUSINESS_SUITABLE.
Title: {context['title']}
Duration: {context['duration_seconds']}s
Language: {context['language']}
Transcript sample: {context['transcript_sample'][:200]}

Respond with JSON: {{"classification": "...", "confidence": 0.0-1.0, "primary_reason": "...", "exclusion_reason": "..."}}"""
    
    # Format the template with context
    prompt = prompt_template.format(
        title=context['title'],
        channel_name=context['channel_name'],
        duration_seconds=context['duration_seconds'],
        duration_minutes=context['duration_seconds']/60,
        language=context['language'],
        tags=context['tags'],
        categories=context['categories'],
        description=context['description'],
        transcript_length=context['transcript_length'],
        transcript_density=context['transcript_density'],
        words_per_minute=context['words_per_minute'],
        transcript_sample=context['transcript_sample']
    )
    
    return prompt


def parse_and_validate_classification(llm_response: str, video_id: str) -> Dict[str, Any]:
    """
    Parse LLM response with comprehensive validation and error handling.
    
    Args:
        llm_response: Raw response from LLM
        video_id: Video ID for logging purposes
        
    Returns:
        Validated classification dictionary
    """
    try:
        # Clean response - remove markdown code blocks if present
        clean_response = llm_response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]  # Remove ```json
        if clean_response.startswith('```'):
            clean_response = clean_response[3:]  # Remove ```
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]  # Remove trailing ```
        clean_response = clean_response.strip()
        
        # Parse JSON response
        result = json.loads(clean_response)
        
        # Validate required fields
        required_fields = ['classification', 'confidence', 'primary_reason', 'exclusion_reason']
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate classification value
        valid_classifications = ['MUSIC_VIDEO', 'BACKGROUND_ONLY', 'BUSINESS_SUITABLE']
        if result['classification'] not in valid_classifications:
            raise ValueError(f"Invalid classification: {result['classification']}")
        
        # Validate and normalize confidence
        try:
            confidence = float(result['confidence'])
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence out of range: {confidence}")
            result['confidence'] = confidence
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid confidence value: {result['confidence']} - {e}")
        
        # Validate exclusion reason logic
        should_exclude = result['classification'] in ['MUSIC_VIDEO', 'BACKGROUND_ONLY']
        if should_exclude and not result['exclusion_reason']:
            result['exclusion_reason'] = 'background_music_only'  # Default fallback
            logger.debug(f"Set default exclusion reason for video {video_id}")
        elif not should_exclude:
            result['exclusion_reason'] = None
        
        # Validate exclusion reasons are in allowed list (language_unsupported handled separately after classification)
        valid_exclusion_reasons = ['background_music_only', None]
        if result['exclusion_reason'] not in valid_exclusion_reasons:
            logger.warning(f"Invalid exclusion reason '{result['exclusion_reason']}' for video {video_id}, defaulting to background_music_only")
            result['exclusion_reason'] = 'background_music_only'
        
        # Validate primary_reason exists and is string
        if not result.get('primary_reason') or not isinstance(result['primary_reason'], str):
            raise ValueError("primary_reason must be a non-empty string")
        
        logger.debug(f"Successfully parsed classification for video {video_id}: "
                    f"{result['classification']} (confidence: {result['confidence']:.2f})")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed for video {video_id}: {e}")
        logger.error(f"Raw LLM response: {llm_response}")
        return _create_fallback_classification(f"JSON parsing failed: {str(e)}")
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Classification validation failed for video {video_id}: {e}")
        logger.error(f"Raw LLM response: {llm_response}")
        return _create_fallback_classification(f"Validation failed: {str(e)}")


def _create_fallback_classification(error_reason: str) -> Dict[str, Any]:
    """
    Create conservative fallback classification for error cases.
    
    Args:
        error_reason: Reason for the fallback
        
    Returns:
        Conservative fallback classification
    """
    return {
        'classification': 'BUSINESS_SUITABLE',  # Conservative - don't exclude on errors
        'confidence': 0.0,
        'primary_reason': f'Classification error: {error_reason}',
        'exclusion_reason': None,
        'error': True
    }


async def classify_video_with_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify video using LLM with comprehensive error handling and logging.
    
    Args:
        context: Video context from prepare_classification_context()
        
    Returns:
        Classification result dictionary
    """
    video_id = context.get('video_id', 'unknown')
    
    try:
        # Build classification prompt
        prompt = build_classification_prompt(context)
        
        # Initialize AI services using existing infrastructure
        config = get_config()
        llm_service = get_gemini_llm_service()
        
        # Create chat request
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a precise video content classifier."),
            ChatMessage(role=ChatRole.USER, content=prompt)
        ]
        
        request = ChatRequest(
            messages=messages,
            model=config.gemini.model,
            temperature=0.3,  # Lower temperature for consistent classification
            max_tokens=500    # Sufficient for JSON response
        )
        
        logger.debug(f"Making LLM classification request for video {video_id}")
        
        # Make LLM call with service wrapper
        result = await llm_service.chat_completion(request)
        
        if result['status'] != 'completed':
            raise Exception(f"LLM service error: {result.get('error', 'Unknown error')}")
        
        response = result['response']
        processing_time = result.get('processing_time_ms', 0)
        
        # Log raw LLM response for debugging
        logger.info(f"Raw LLM response for video {video_id}:")
        logger.info(f"Response: {response.choices[0].message.content}")
        logger.info(f"Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'unknown'}")
        
        # Parse and validate response
        classification = parse_and_validate_classification(response.choices[0].message.content, video_id)
        
        # Add processing metadata
        classification['processing_time_ms'] = processing_time
        classification['tokens_used'] = response.usage.total_tokens if hasattr(response, 'usage') else 0
        
        # Log successful classification
        logger.info(f"Video {video_id} classified as {classification['classification']} "
                   f"(confidence: {classification['confidence']:.2f}) - {classification['primary_reason']}")
        
        return classification
        
    except Exception as e:
        # Comprehensive error logging
        logger.error(f"LLM classification failed for video {video_id}: {str(e)}")
        logger.error(f"Context summary: title='{context.get('title', 'N/A')}', "
                    f"duration={context.get('duration_seconds', 'N/A')}s, "
                    f"transcript_length={context.get('transcript_length', 'N/A')}")
        
        # Return conservative fallback
        return _create_fallback_classification(str(e))


@shared_task(bind=True, max_retries=2, retry_delay=10)
def classify_and_exclude_video_llm(self, previous_result, url_request_id):
    """
    Classify video and make exclusion decisions before embedding stage.
    
    This is the main entry point for the LLM-based classification system.
    It runs after summary generation and before the embedding stage.
    
    Args:
        self: Celery task instance (for retry functionality)
        previous_result: Result from previous task in chain
        url_request_id: ID of URLRequest being processed
        
    Returns:
        Dict with classification results and processing status
    """
    # Check if video was excluded in previous stage
    if previous_result and previous_result.get('excluded'):
        logger.info(f"Video was excluded in previous stage: {previous_result.get('exclusion_reason')}")
        # Return immediately without processing classification
        return {
            'video_id': previous_result.get('video_id'),
            'excluded': True,
            'exclusion_reason': previous_result.get('exclusion_reason'),
            'skip_reason': 'excluded_in_previous_stage',
            'status': 'completed'
        }
    
    try:
        # Get video data with validation
        try:
            video_metadata = VideoMetadata.objects.get(url_request__id=url_request_id)
        except VideoMetadata.DoesNotExist:
            logger.error(f"VideoMetadata not found for URL request {url_request_id}")
            return {
                'status': 'failed',
                'error': 'VideoMetadata not found',
                'url_request_id': url_request_id
            }
        
        # Validate transcript exists
        if not hasattr(video_metadata, 'video_transcript') or not video_metadata.video_transcript:
            logger.error(f"No transcript found for video {video_metadata.video_id}")
            return {
                'video_id': video_metadata.video_id,
                'status': 'failed',
                'error': 'No transcript available for classification'
            }
        
        transcript = video_metadata.video_transcript
        
        # Log classification start
        logger.info(f"Starting LLM classification for video {video_metadata.video_id}: "
                   f"'{video_metadata.title[:50]}...' ({video_metadata.duration}s)")
        
        # Prepare context for classification
        try:
            context = prepare_classification_context(video_metadata, transcript)
        except ValueError as e:
            logger.error(f"Context preparation failed for video {video_metadata.video_id}: {e}")
            return {
                'video_id': video_metadata.video_id,
                'status': 'failed',
                'error': f'Context preparation failed: {str(e)}'
            }
        
        # Get LLM classification (async function in sync task context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            classification = loop.run_until_complete(classify_video_with_llm(context))
        finally:
            loop.close()
        
        # Make LLM-based exclusion decision
        llm_should_exclude = (
            classification['classification'] in ['MUSIC_VIDEO', 'BACKGROUND_ONLY'] and
            classification['confidence'] >= 0.7 and
            not classification.get('error', False)
        )
        
        # Perform final language check after classification
        language_should_exclude = False
        exclusion_reason = None
        
        if llm_should_exclude:
            # LLM decided to exclude - use LLM's reason
            exclusion_reason = classification['exclusion_reason'] or 'background_music_only'
            should_exclude = True
            logger.info(f"LLM exclusion decision for {video_metadata.video_id}: {exclusion_reason}")
        else:
            # LLM said BUSINESS_SUITABLE - check language as final gate
            try:
                language_should_exclude, language_reason = should_exclude_for_language(video_metadata, transcript)
                if language_should_exclude:
                    exclusion_reason = 'language_unsupported'
                    should_exclude = True
                    logger.info(f"Language exclusion override for {video_metadata.video_id}: LLM said BUSINESS_SUITABLE but language check failed ({language_reason})")
                else:
                    should_exclude = False
                    logger.info(f"Final decision for {video_metadata.video_id}: RETAINED (LLM: BUSINESS_SUITABLE, Language: {language_reason})")
            except Exception as e:
                logger.error(f"Language exclusion check failed for {video_metadata.video_id}: {e}")
                # Be conservative - if language check fails, don't exclude based on language
                should_exclude = False
        
        if should_exclude:
            # Add to exclusion table
            logger.info(f"🚫 EXCLUDING video {video_metadata.video_id} for reason: {exclusion_reason}")
            logger.info(f"   Classification: {classification['classification']} (confidence: {classification['confidence']:.2f})")
            logger.info(f"   URL: {video_metadata.url_request.url}")
            
            try:
                added = add_video_to_exclusion_table(
                    video_metadata.url_request.url,
                    exclusion_reason
                )
                
                if added:
                    logger.info(f"✅ Successfully added video {video_metadata.video_id} to exclusion table: {exclusion_reason}")
                else:
                    logger.warning(f"⚠️  Video {video_metadata.video_id} was NOT added to exclusion table (likely already exists)")
                    
            except Exception as e:
                logger.error(f"❌ Error adding video {video_metadata.video_id} to exclusion table: {e}")
                logger.exception("Full traceback:")
            
            # Mark URLRequest as failed for business logic exclusion
            try:
                old_status = video_metadata.url_request.status
                video_metadata.url_request.status = 'failed'
                video_metadata.url_request.failure_reason = 'excluded'
                video_metadata.url_request.save()
                logger.info(f"✅ Updated URLRequest status: {old_status} → failed for video {video_metadata.video_id}")
            except Exception as e:
                logger.error(f"❌ Failed to update URLRequest status for video {video_metadata.video_id}: {e}")
                logger.exception("Full traceback:")
                
        else:
            # Log why not excluded
            if classification.get('error', False):
                logger.warning(f"NOT EXCLUDED due to classification error: {video_metadata.video_id}")
            elif classification['confidence'] < 0.7:
                logger.warning(f"NOT EXCLUDED due to low confidence ({classification['confidence']:.2f}): "
                              f"{video_metadata.video_id}")
            else:
                logger.info(f"RETAINED video {video_metadata.video_id}: {classification['classification']} "
                           f"(confidence: {classification['confidence']:.2f})")
        
        # Return comprehensive result for pipeline
        result = {
            'video_id': video_metadata.video_id,
            'classification': classification['classification'],
            'confidence': classification['confidence'],
            'excluded': should_exclude,
            'exclusion_reason': classification.get('exclusion_reason'),
            'reasoning': classification['primary_reason'],
            'processing_time_ms': classification.get('processing_time_ms', 0),
            'tokens_used': classification.get('tokens_used', 0),
            'status': 'completed'
        }
        
        logger.debug(f"Classification completed for video {video_metadata.video_id}: {result}")
        return result
        
    except Exception as e:
        # Comprehensive error logging with context
        logger.error(f"Classification task failed for URL request {url_request_id}: {str(e)}")
        logger.exception("Full traceback:")
        
        # Retry logic for transient errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying classification for URL request {url_request_id} "
                       f"(attempt {self.request.retries + 1}/{self.max_retries})")
            
            # Exponential backoff for retries
            countdown = 10 * (2 ** self.request.retries)
            raise self.retry(countdown=countdown, exc=e)
        
        # Final failure - conservative approach (don't exclude)
        logger.error(f"Classification permanently failed for URL request {url_request_id} "
                    f"after {self.max_retries} retries")
        
        return {
            'video_id': 'unknown',
            'classification': 'BUSINESS_SUITABLE',  # Conservative fallback
            'confidence': 0.0,
            'excluded': False,
            'status': 'failed',
            'error': str(e),
            'retries_exhausted': True
        }