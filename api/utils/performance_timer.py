import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, List

logger = logging.getLogger(__name__)

class RAGPerformanceTracker:
    """
    Performance tracker for RAG pipeline stages.
    Provides detailed timing analysis to identify bottlenecks.
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.current_request_timings: Dict[str, float] = {}
        self.request_start_time = time.time()
    
    @asynccontextmanager
    async def time_stage(self, stage_name: str):
        """Context manager for timing a specific stage"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            self.current_request_timings[stage_name] = elapsed
            logger.debug(f"Stage {stage_name}: {elapsed:.2f}ms")
    
    def time_stage_sync(self, stage_name: str, start_time: float):
        """Manual timing for synchronous operations"""
        elapsed = (time.time() - start_time) * 1000
        self.current_request_timings[stage_name] = elapsed
        logger.debug(f"Stage {stage_name}: {elapsed:.2f}ms")
        return elapsed
    
    def log_request_summary(self, question: str):
        """Log comprehensive timing summary for the request"""
        total_request_time = (time.time() - self.request_start_time) * 1000
        total_measured_time = sum(self.current_request_timings.values())
        
        # Log timing summary for debugging
        logger.debug(f"RAG timing summary for '{question[:50]}{'...' if len(question) > 50 else ''}'")
        logger.debug(f"Total request time: {total_request_time:.2f}ms")
        
        # Sort stages by time descending to show biggest bottlenecks first
        sorted_stages = sorted(self.current_request_timings.items(), key=lambda x: x[1], reverse=True)
        
        for stage, time_ms in sorted_stages:
            percentage = (time_ms / total_measured_time) * 100 if total_measured_time > 0 else 0
            logger.debug(f"  {stage}: {time_ms:.2f}ms ({percentage:.1f}%)")
        
        # Performance assessment
        if total_request_time < 2000:
            status = "EXCELLENT"
        elif total_request_time < 4000:
            status = "GOOD"
        elif total_request_time < 6000:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS OPTIMIZATION"
            
        logger.debug(f"Performance: {total_request_time:.0f}ms - {status}")
        
        # Bottleneck identification
        if self.current_request_timings:
            max_stage = max(self.current_request_timings.items(), key=lambda x: x[1])
            max_percentage = (max_stage[1] / total_measured_time) * 100 if total_measured_time > 0 else 0
            if max_percentage > 50:
                logger.debug(f"Primary bottleneck: {max_stage[0]} ({max_percentage:.1f}% of time)")
        
        # Reset for next request
        self.current_request_timings.clear()
        self.request_start_time = time.time()
    
    def get_stage_timings(self) -> Dict[str, float]:
        """Get current stage timings (useful for APIs)"""
        return self.current_request_timings.copy() 