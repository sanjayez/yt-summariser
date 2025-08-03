"""
Response Service for API response formatting and processing.
Handles response compression, formatting, and standardization.
"""
from typing import List, Dict, Any, Optional
from telemetry import get_logger


class ResponseService:
    """
    Handles response formatting and processing for the API layer.
    
    This service provides:
    - Context compression for LLM efficiency
    - Response formatting and standardization
    - Source formatting for search results
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def compress_context(self, results: List[Dict[str, Any]], max_chars: int = 600) -> str:
        """
        Compress context to reduce LLM processing time while preserving meaningful content.
        Expected savings: ~500ms-1s due to optimized token count.
        
        This function was extracted from api/views.py lines 21-66.
        
        Args:
            results: List of search results with text, score, type, and metadata
            max_chars: Maximum characters to include in compressed context
            
        Returns:
            Compressed context string formatted for LLM consumption
        """
        compressed = []
        char_count = 0
        
        # Sort by score and process highest relevance first
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        for result in sorted_results:
            # Get meaningful text from the segment
            text = result.get('text', '')
            
            # Clean and extract meaningful content
            if "(from:" in text:
                # Remove the truncated suffix like "(from: TOP 10 HIDDEN..."
                text = text.split("(from:")[0].strip()
            
            # Ensure minimum meaningful length
            if len(text) < 20:
                continue
                
            # Allow much longer segments to preserve important content (increased from 250 to 1100)
            if len(text) > 1100:
                text = text[:1097] + "..."
            
            # Format with timestamp for context
            metadata = result.get('metadata', {})
            if result.get('type') == 'segment' and 'start_time' in metadata:
                timestamp = metadata['start_time']
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                segment = f"[{time_str}] {text}"
            else:
                segment = text
            
            # Check if adding this segment would exceed limit
            if char_count + len(segment) + 2 > max_chars:  # +2 for \n
                break
                
            compressed.append(segment)
            char_count += len(segment) + 2
        
        result_text = "\n".join(compressed)
        self.logger.debug(f"Compressed {len(results)} results to {len(result_text)} chars")
        
        return result_text
    
    def format_video_metadata(self, video_metadata) -> Dict[str, Any]:
        """
        Format video metadata for API responses.
        
        Args:
            video_metadata: VideoMetadata model instance
            
        Returns:
            Formatted metadata dictionary
        """
        if not video_metadata:
            return {}
        
        # Format description with truncation
        description = video_metadata.description
        if description and len(description) > 200:
            description = description[:200] + '...'
        
        return {
            'video_id': video_metadata.video_id,
            'title': video_metadata.title,
            'description': description,
            'duration': video_metadata.duration,
            'duration_string': video_metadata.duration_string,
            'channel_name': video_metadata.channel_name,
            'view_count': video_metadata.view_count,
            'like_count': video_metadata.like_count,
            'upload_date': video_metadata.upload_date.isoformat() if video_metadata.upload_date else None,
            'language': video_metadata.language,
            'tags': video_metadata.tags[:10] if video_metadata.tags else [],  # Limit tags
            'youtube_url': str(video_metadata.webpage_url),
            'thumbnail': str(video_metadata.thumbnail) if video_metadata.thumbnail else None
        }
    
    def format_search_sources(self, results: List[Dict[str, Any]], video_metadata=None) -> List[Dict[str, Any]]:
        """
        Format search results into standardized source format.
        
        Args:
            results: Raw search results
            video_metadata: Video metadata for URL context
            
        Returns:
            List of formatted source dictionaries
        """
        formatted_sources = []
        
        for i, result in enumerate(results):
            result_type = result.get('type')
            metadata = result.get('metadata', {})
            confidence = result.get('score', 0.0)
            
            if result_type == 'segment':
                # Handle segments with precise timestamps
                timestamp = metadata.get('start_time', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                # Clean text for sources (remove timestamp if present)
                clean_text = result.get('text', '')
                if clean_text.startswith(f"[{time_str}]"):
                    clean_text = clean_text.replace(f"[{time_str}]", "").strip()
                
                formatted_sources.append({
                    'type': 'segment',
                    'timestamp': time_str,
                    'text': clean_text,
                    'youtube_url': metadata.get('youtube_url', str(video_metadata.webpage_url) if video_metadata else ''),
                    'confidence': confidence
                })
                
            elif result_type == 'transcript_chunk':
                # Handle transcript chunks (no specific timestamps, but show as chunks)
                formatted_sources.append({
                    'type': 'chunk',
                    'timestamp': 'Multiple',  # Chunks span multiple timestamps
                    'text': result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', ''),
                    'youtube_url': str(video_metadata.webpage_url) if video_metadata else '',
                    'confidence': confidence
                })
                
            else:
                # Handle other source types (summary, metadata, etc.)
                formatted_sources.append({
                    'type': result_type or 'transcript',
                    'timestamp': 'Unknown',
                    'text': result.get('text', ''),
                    'youtube_url': str(video_metadata.webpage_url) if video_metadata else '',
                    'confidence': confidence
                })
        
        return formatted_sources
    
    def format_error_response(self, error_type: str, message: str, status: str = "error", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format standardized error responses.
        
        Args:
            error_type: Type or category of error
            message: Detailed error message
            status: Error status code
            details: Additional error details
            
        Returns:
            Formatted error response dictionary
        """
        response = {
            'error': error_type,
            'status': status,
            'message': message
        }
        
        if details:
            response['details'] = details
        
        return response
    
    def calculate_confidence_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score from search results.
        
        Args:
            results: List of search results with scores
            
        Returns:
            Average confidence score between 0.0 and 1.0
        """
        if not results:
            return 0.0
        
        scores = [result.get('score', 0.0) for result in results]
        valid_scores = [score for score in scores if isinstance(score, (int, float))]
        
        if not valid_scores:
            return 0.0
        
        avg_confidence = sum(valid_scores) / len(valid_scores)
        return round(min(max(avg_confidence, 0.0), 1.0), 2)  # Clamp between 0 and 1
    
    def format_timing_info(self, stage_timings: Dict[str, float], total_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Format timing information for responses.
        
        Args:
            stage_timings: Dictionary of stage names to timing values
            total_time: Total operation time
            
        Returns:
            Formatted timing information
        """
        if not stage_timings:
            return {}
        
        total_measured = sum(stage_timings.values())
        
        timing_info = {
            'stage_timings': stage_timings,
            'total_time_ms': round(total_time or total_measured, 2),
            'measured_stages': len(stage_timings)
        }
        
        if total_time:
            timing_info['note'] = 'Total RAG pipeline execution time'
        
        return timing_info