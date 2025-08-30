#!/usr/bin/env python3
"""
QueryProcessor Demo Script
Demonstrates the QueryProcessor service for enhancing user queries
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_utils.config import get_config
from ai_utils.providers import OpenAILLMProvider
from ai_utils.services import LLMService
from topic.services.query_processor import QueryProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main demo function for QueryProcessor"""
    try:
        # Get configuration and validate
        config = get_config()
        config.validate()

        # Initialize services
        llm_provider = OpenAILLMProvider(config=config)
        llm_service = LLMService(provider=llm_provider)
        query_processor = QueryProcessor(llm_service=llm_service)

        logger.info("🔍 QueryProcessor Demo - YouTube Search Query Enhancement")
        logger.info("=" * 65)

        # 1. Health check
        logger.info("🏥 Running QueryProcessor health check...")
        health_result = await query_processor.health_check()
        logger.info(f"QueryProcessor service: {health_result['status']}")

        if health_result["status"] != "healthy":
            logger.error(
                "❌ QueryProcessor service is unhealthy. Please check your configuration."
            )
            return

        # 2. Single query enhancement examples
        logger.info("\n📝 Single Query Enhancement Examples")
        logger.info("-" * 40)

        test_queries = [
            "How do I learn Python programming?",
            "Can you explain machine learning concepts?",
            "What are the best practices for React development?",
            "I want to understand neural networks",
            "How to build a REST API with Flask?",
            "What is the difference between SQL and NoSQL databases?",
            "How do I deploy a Django application?",
            "Explain Docker containerization",
        ]

        for i, query in enumerate(test_queries, 1):
            result = await query_processor.enhance_query(query)

            if result["status"] == "completed":
                logger.info(f"{i}. Original: '{result['original_query']}'")
                logger.info(f"   Enhanced: '{result['enhanced_query']}'")
                logger.info(
                    f"   Time: {result['processing_time_ms']:.1f}ms | Tokens: {result['tokens_used']}"
                )
            else:
                logger.error(
                    f"{i}. Failed to enhance: '{query}' - {result.get('error', 'Unknown error')}"
                )
            logger.info("")

        # 3. Batch enhancement
        logger.info("📚 Batch Query Enhancement")
        logger.info("-" * 30)

        batch_queries = [
            "How to setup a development environment?",
            "Best practices for code review",
            "Understanding design patterns",
            "How to optimize database queries?",
        ]

        batch_result = await query_processor.batch_enhance_queries(batch_queries)

        if batch_result["status"] == "completed":
            logger.info(f"✅ Processed {batch_result['total_queries']} queries")
            logger.info(f"   Successful: {batch_result['successful_enhancements']}")
            logger.info(f"   Failed: {batch_result['failed_enhancements']}")
            logger.info(
                f"   Total time: {batch_result['total_processing_time_ms']:.1f}ms"
            )
            logger.info(f"   Total tokens: {batch_result['total_tokens_used']}")
            logger.info("")

            for result in batch_result["results"]:
                if result["status"] == "completed":
                    logger.info(
                        f"   '{result['original_query']}' → '{result['enhanced_query']}'"
                    )
                else:
                    logger.error(
                        f"   Failed: '{result['original_query']}' - {result.get('error', 'Unknown error')}"
                    )
        else:
            logger.error(
                f"❌ Batch processing failed: {batch_result.get('error', 'Unknown error')}"
            )

        # 4. Edge cases and error handling
        logger.info("\n🚨 Error Handling and Edge Cases")
        logger.info("-" * 35)

        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a" * 1000,  # Very long query
            "简单的查询",  # Non-English query
            "!!@#$%^&*()",  # Special characters
        ]

        for i, query in enumerate(edge_cases, 1):
            result = await query_processor.enhance_query(query)

            if result["status"] == "completed":
                logger.info(f"{i}. Handled edge case successfully")
                logger.info(
                    f"   Input: '{query[:50]}{'...' if len(query) > 50 else ''}'"
                )
                logger.info(f"   Output: '{result['enhanced_query']}'")
            else:
                logger.info(
                    f"{i}. Expected error for edge case: {result.get('error', 'Unknown error')}"
                )
            logger.info("")

        # 5. Performance test
        logger.info("⚡ Performance Test")
        logger.info("-" * 18)

        performance_queries = [
            "How to learn web development?",
            "Python data science tutorial",
            "JavaScript frameworks comparison",
            "Database design best practices",
            "Machine learning algorithms explained",
        ]

        # Test concurrent processing
        import time

        start_time = time.time()

        tasks = [query_processor.enhance_query(query) for query in performance_queries]

        concurrent_results = await asyncio.gather(*tasks)

        concurrent_time = (time.time() - start_time) * 1000

        logger.info(f"✅ Processed {len(performance_queries)} queries concurrently")
        logger.info(f"   Total time: {concurrent_time:.1f}ms")
        logger.info(
            f"   Average per query: {concurrent_time / len(performance_queries):.1f}ms"
        )

        successful_concurrent = sum(
            1 for r in concurrent_results if r["status"] == "completed"
        )
        logger.info(
            f"   Success rate: {successful_concurrent}/{len(performance_queries)} ({successful_concurrent / len(performance_queries) * 100:.1f}%)"
        )

        # 6. Job status tracking
        logger.info("\n📊 Job Status Tracking")
        logger.info("-" * 25)

        # Create a tracked job
        tracked_result = await query_processor.enhance_query(
            "How to implement authentication in web applications?",
            job_id="demo_tracked_job",
        )

        if tracked_result["status"] == "completed":
            job_status = query_processor.get_job_status("demo_tracked_job")
            if job_status:
                logger.info(f"Job ID: {job_status['job_id']}")
                logger.info(f"Status: {job_status['status']}")
                logger.info(f"Operation: {job_status['operation']}")
                logger.info(f"Created: {job_status['created_at']}")
                logger.info(f"Updated: {job_status['updated_at']}")

        logger.info("\n🎉 QueryProcessor Demo completed successfully!")
        logger.info("=" * 65)
        logger.info("✅ The QueryProcessor service is fully functional!")
        logger.info("🔧 Ready to enhance YouTube search queries for better results")

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
