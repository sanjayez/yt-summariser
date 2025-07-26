"""
Video Content Embedding Task
Implements 4-layer embedding strategy for comprehensive video content search:
1. Metadata embedding (video-level context)
2. Summary embedding (key insights) 
3. Transcript chunks embedding (comprehensive search)
4. Segment embeddings (precise timestamp navigation)
"""

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from celery_progress.backend import ProgressRecorder
from django.db import transaction
import logging
import asyncio
from typing import List, Dict, Any

from api.models import URLRequestTable
from ..models import VideoMetadata, VideoTranscript, TranscriptSegment
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)
from ..text_utils.chunking import (
    chunk_transcript_text,
    format_metadata_for_embedding,
    format_summary_for_embedding,
    format_segment_for_embedding,
    validate_embedding_text,
    prepare_batch_embeddings
)
from ai_utils.services.embedding_service import EmbeddingService
from ai_utils.services.vector_service import VectorService
from ai_utils.models import VectorDocument

logger = logging.getLogger(__name__)

def embed_video_content_sync(video_metadata: VideoMetadata, transcript: VideoTranscript) -> Dict[str, Any]:
    """
    Synchronous helper function to embed all video content using 4-layer strategy.
    Returns embedding statistics and results.
    """
    try:
        # Initialize services with sync providers
        from ai_utils.config import get_config
        from ai_utils.providers.openai_embeddings import OpenAIEmbeddingProvider
        from ai_utils.providers.pinecone_store import PineconeVectorStoreProvider
        from ai_utils.services.embedding_service import EmbeddingService
        from ai_utils.services.vector_service import VectorService
        
        config = get_config()
        embedding_provider = OpenAIEmbeddingProvider(config)
        vector_provider = PineconeVectorStoreProvider(config)
        
        embedding_service = EmbeddingService(embedding_provider)
        vector_service = VectorService(vector_provider)
        
        video_id = video_metadata.video_id
        embedding_items = []
        
        # Layer 1: Metadata embedding
        logger.info(f"Preparing metadata embedding for video {video_id}")
        metadata_text = format_metadata_for_embedding(video_metadata)
        metadata_text = validate_embedding_text(metadata_text)
        
        embedding_items.append({
            'id': f"meta_{video_id}",
            'text': metadata_text,
            'metadata': {
                'type': 'metadata',
                'video_id': video_id,
                'title': video_metadata.title or 'Unknown',
                'channel': video_metadata.channel_name or 'Unknown',
                'duration': video_metadata.duration or 0
            }
        })
        
        # Layer 2: Summary embedding (only if summary exists)
        if transcript.summary and transcript.summary.strip():
            logger.info(f"Preparing summary embedding for video {video_id}")
            summary_text = format_summary_for_embedding(transcript.summary, transcript.key_points)
            summary_text = validate_embedding_text(summary_text)
            
            embedding_items.append({
                'id': f"summary_{video_id}",
                'text': summary_text,
                'metadata': {
                    'type': 'summary',
                    'video_id': video_id,
                    'title': video_metadata.title or 'Unknown',
                    'key_points_count': len(transcript.key_points) if transcript.key_points else 0
                }
            })
        
        # Layer 3: Transcript chunks embedding
        if transcript.transcript_text and transcript.transcript_text.strip():
            logger.info(f"Preparing transcript chunks embedding for video {video_id}")
            transcript_chunks = chunk_transcript_text(transcript.transcript_text, chunk_size=1000, chunk_overlap=200)
            
            for i, chunk in enumerate(transcript_chunks):
                chunk_text = validate_embedding_text(chunk)
                embedding_items.append({
                    'id': f"transcript_{video_id}_{i}",
                    'text': chunk_text,
                    'metadata': {
                        'type': 'transcript_chunk',
                        'video_id': video_id,
                        'chunk_index': i,
                        'title': video_metadata.title or 'Unknown'
                    }
                })
        
        # Layer 4: Segment embeddings
        segments = transcript.segments.all()
        if segments.exists():
            logger.info(f"Preparing segment embeddings for video {video_id}: {segments.count()} segments")
            
            for segment in segments:
                segment_text = format_segment_for_embedding(segment)
                segment_text = validate_embedding_text(segment_text)
                
                embedding_items.append({
                    'id': segment.segment_id,
                    'text': segment_text,
                    'metadata': {
                        'type': 'segment',
                        'video_id': video_id,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'sequence_number': segment.sequence_number,
                        'title': video_metadata.title or 'Unknown',
                        'youtube_url': segment.get_youtube_url_with_timestamp()
                    }
                })
        
        logger.info(f"Prepared {len(embedding_items)} items for embedding")
        
        # Batch process embeddings using custom approach
        batches = prepare_batch_embeddings(embedding_items, batch_size=15)
        total_embedded = 0
        failed_embeddings = []
        
        for batch_idx, batch in enumerate(batches):
            try:
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} items")
                
                # Extract texts for embedding
                texts = [item['text'] for item in batch]
                
                # Generate embeddings for the batch
                try:
                    import asyncio
                    
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        # Get embeddings using the embedding service
                        embeddings = loop.run_until_complete(
                            embedding_service.embed_texts_with_batching(texts)
                        )
                        
                        if len(embeddings) != len(texts):
                            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                        
                        # Create VectorDocuments with our specific IDs and the generated embeddings
                        vector_documents = []
                        for i, (item, embedding) in enumerate(zip(batch, embeddings)):
                            vector_doc = VectorDocument(
                                id=item['id'],  # Use our specific ID
                                text=item['text'],
                                embedding=embedding,  # Use the generated embedding
                                metadata=item['metadata']
                            )
                            vector_documents.append(vector_doc)
                        
                        # Upsert the documents with embeddings
                        result = loop.run_until_complete(
                            vector_service.upsert_documents(
                                documents=vector_documents,
                                job_id=f"embed_batch_{video_id}_{batch_idx}"
                            )
                        )
                        
                        if result and result.get('upserted_count', 0) > 0:
                            total_embedded += result['upserted_count']
                            logger.info(f"Successfully embedded batch {batch_idx + 1}: {result['upserted_count']} items")
                        else:
                            logger.warning(f"Failed to embed batch {batch_idx + 1}")
                            failed_embeddings.extend([item['id'] for item in batch])
                            
                    finally:
                        loop.close()
                        
                except Exception as embed_error:
                    logger.error(f"Error embedding batch {batch_idx + 1}: {embed_error}")
                    failed_embeddings.extend([item['id'] for item in batch])
                    continue
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                failed_embeddings.extend([item['id'] for item in batch])
                continue
        
        # Update embedding status in database
        with transaction.atomic():
            # Mark metadata as embedded
            video_metadata.is_embedded = True
            video_metadata.save()
            
            # Mark segments as embedded (only successful ones)
            if segments.exists():
                successful_segment_ids = [
                    item['id'] for item in embedding_items 
                    if item['metadata']['type'] == 'segment' and item['id'] not in failed_embeddings
                ]
                
                if successful_segment_ids:
                    segments.filter(segment_id__in=successful_segment_ids).update(is_embedded=True)
        
        result = {
            'total_items': len(embedding_items),
            'total_embedded': total_embedded,
            'failed_embeddings': len(failed_embeddings),
            'batches_processed': len(batches),
            'metadata_embedded': True,
            'summary_embedded': transcript.summary is not None and transcript.summary.strip(),
            'transcript_chunks_embedded': len([item for item in embedding_items if item['metadata']['type'] == 'transcript_chunk']),
            'segments_embedded': len([item for item in embedding_items if item['metadata']['type'] == 'segment']) - len([item for item in failed_embeddings if item.startswith(video_id)]),
            'video_id': video_id
        }
        
        logger.info(f"Embedding completed for video {video_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in sync embedding process: {e}")
        raise

@shared_task(bind=True,
             name='video_processor.embed_video_content',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['embedding_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['embedding_hard_limit'],
             max_retries=YOUTUBE_CONFIG['RETRY_CONFIG']['embedding']['max_retries'],
             default_retry_delay=YOUTUBE_CONFIG['RETRY_CONFIG']['embedding']['countdown'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['embedding']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['embedding']['jitter'])
@idempotent_task
def embed_video_content(self, summary_result, url_request_id):
    """
    Embed video content using 4-layer strategy after summary generation.
    
    Args:
        summary_result (dict): Result from previous summary generation task
        url_request_id (int): ID of the URLRequestTable to process
        
    Returns:
        dict: Embedding results with total items embedded and processing stats
        
    Raises:
        Exception: If embedding process fails after retries
    """
    url_request = None
    video_metadata = None
    transcript = None
    
    try:
        # Set initial progress
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(0, 100, "Generating embeddings")
        update_task_progress(self, TASK_STATES.get('EMBEDDING_CONTENT', 'Embedding Content'), 10)
        
        # Get video metadata and transcript
        url_request = URLRequestTable.objects.select_related(
            'video_metadata'
        ).get(id=url_request_id)
        
        # Check if video is excluded (skip embedding for excluded videos)
        if url_request.failure_reason == 'excluded':
            video_id = getattr(url_request, 'video_metadata', None)
            video_id = getattr(video_id, 'video_id', 'unknown') if video_id else 'unknown'
            logger.info(f"Skipping embedding for excluded video {video_id} (reason: {url_request.failure_reason})")
            return {
                'skipped': True, 
                'reason': 'excluded',
                'video_id': video_id,
                'total_items': 0,
                'total_embedded': 0,
                'already_embedded': False
            }
        
        if not hasattr(url_request, 'video_metadata'):
            raise ValueError("VideoMetadata not found for this request")
        
        video_metadata = url_request.video_metadata
        
        if not hasattr(video_metadata, 'video_transcript'):
            raise ValueError("VideoTranscript not found for this request")
        
        transcript = video_metadata.video_transcript
        
        logger.info(f"Starting embedding process for video {video_metadata.video_id}")
        
        update_task_progress(self, TASK_STATES.get('EMBEDDING_CONTENT', 'Embedding Content'), 30)
        
        # Check if already embedded (idempotency)
        if video_metadata.is_embedded:
            logger.info(f"Video {video_metadata.video_id} already embedded, skipping")
            return {
                'total_items': 0,
                'total_embedded': 0,
                'already_embedded': True,
                'video_id': video_metadata.video_id
            }
        
        # Perform embedding using synchronous function
        embedding_result = embed_video_content_sync(video_metadata, transcript)
        
        update_task_progress(self, TASK_STATES.get('EMBEDDING_CONTENT', 'Embedding Content'), 90)
        
        # Log final results
        logger.info(f"Embedding completed for video {video_metadata.video_id}: "
                   f"{embedding_result['total_embedded']}/{embedding_result['total_items']} items embedded")
        
        update_task_progress(self, TASK_STATES.get('EMBEDDING_CONTENT', 'Embedding Content'), 100)
        
        # Set final progress
        progress_recorder.set_progress(100, 100, "Embeddings complete")
        
        return embedding_result
        
    except SoftTimeLimitExceeded:
        # Task is approaching timeout - save status and exit gracefully
        logger.warning(f"Embedding generation soft timeout reached for request {url_request_id}")
        
        try:
            # Mark embedding as failed due to timeout
            if video_metadata:
                video_metadata.is_embedded = False
                video_metadata.save()
                logger.error(f"Marked embedding generation as failed due to timeout: {video_metadata.video_id}")
                
        except Exception as cleanup_error:
            logger.error(f"Failed to update embedding status during timeout cleanup: {cleanup_error}")
        
        # Re-raise with specific timeout message
        raise Exception(f"Embedding generation timeout for request {url_request_id}")
        
    except Exception as e:
        logger.error(f"Embedding failed for request {url_request_id}: {e}")
        
        # Mark as failed but don't stop the chain
        try:
            with transaction.atomic():
                if not url_request:
                    url_request = URLRequestTable.objects.select_related(
                    'video_metadata'
                ).get(id=url_request_id)
                
                if hasattr(url_request, 'video_metadata'):
                    video_metadata = url_request.video_metadata
                    video_metadata.is_embedded = False
                    video_metadata.save()
                    
        except Exception as db_error:
            logger.error(f"Failed to update embedding status: {db_error}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('embed_video_content', self.request.id, [url_request_id], {}, e)
        
        # Return error result but don't break the chain
        return {
            'total_items': 0,
            'total_embedded': 0,
            'failed_embeddings': 0,
            'video_id': 'Unknown',
            'error': str(e)
        } 