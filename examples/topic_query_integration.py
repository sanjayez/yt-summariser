#!/usr/bin/env python3
"""
Topic Query Integration Example
Shows how to integrate QueryProcessor into topic app workflows
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
from topic.services import QueryProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TopicQueryService:
    """
    Example service showing how to integrate QueryProcessor
    into topic-based workflows
    """

    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor

    async def process_topic_query(
        self, user_query: str, topic_context: str = None
    ) -> dict:
        """
        Process a topic-based query with optional context

        Args:
            user_query: The user's original query
            topic_context: Optional topic context to enhance the query

        Returns:
            Enhanced query result with topic context
        """
        try:
            # Enhance the basic query
            enhanced_result = await self.query_processor.enhance_query(user_query)

            if enhanced_result["status"] != "completed":
                return enhanced_result

            # If topic context is provided, further enhance the query
            if topic_context:
                contextual_query = (
                    f"{enhanced_result['enhanced_query']} {topic_context}"
                )

                # Re-enhance with context
                final_result = await self.query_processor.enhance_query(
                    contextual_query
                )

                if final_result["status"] == "completed":
                    return {
                        "original_query": user_query,
                        "enhanced_query": enhanced_result["enhanced_query"],
                        "contextual_query": final_result["enhanced_query"],
                        "topic_context": topic_context,
                        "status": "completed",
                        "processing_time_ms": enhanced_result["processing_time_ms"]
                        + final_result["processing_time_ms"],
                        "tokens_used": enhanced_result["tokens_used"]
                        + final_result["tokens_used"],
                    }
                else:
                    # Fallback to non-contextual result
                    return {
                        "original_query": user_query,
                        "enhanced_query": enhanced_result["enhanced_query"],
                        "contextual_query": enhanced_result["enhanced_query"],
                        "topic_context": topic_context,
                        "status": "completed",
                        "processing_time_ms": enhanced_result["processing_time_ms"],
                        "tokens_used": enhanced_result["tokens_used"],
                        "context_enhancement_failed": True,
                    }

            return {
                "original_query": user_query,
                "enhanced_query": enhanced_result["enhanced_query"],
                "contextual_query": enhanced_result["enhanced_query"],
                "topic_context": None,
                "status": "completed",
                "processing_time_ms": enhanced_result["processing_time_ms"],
                "tokens_used": enhanced_result["tokens_used"],
            }

        except Exception as e:
            logger.error(f"Error in topic query processing: {str(e)}")
            return {
                "original_query": user_query,
                "enhanced_query": user_query,
                "contextual_query": user_query,
                "topic_context": topic_context,
                "status": "failed",
                "error": str(e),
            }

    async def batch_process_topic_queries(self, query_topic_pairs: list) -> dict:
        """
        Process multiple topic queries in batch

        Args:
            query_topic_pairs: List of (query, topic_context) tuples

        Returns:
            Batch processing results
        """
        try:
            results = []

            for i, (query, topic_context) in enumerate(query_topic_pairs):
                result = await self.process_topic_query(query, topic_context)
                result["index"] = i
                results.append(result)

            successful_count = sum(1 for r in results if r["status"] == "completed")
            total_time = sum(r.get("processing_time_ms", 0) for r in results)
            total_tokens = sum(r.get("tokens_used", 0) for r in results)

            return {
                "results": results,
                "status": "completed",
                "total_queries": len(query_topic_pairs),
                "successful_queries": successful_count,
                "failed_queries": len(query_topic_pairs) - successful_count,
                "total_processing_time_ms": total_time,
                "total_tokens_used": total_tokens,
            }

        except Exception as e:
            logger.error(f"Error in batch topic query processing: {str(e)}")
            return {"status": "failed", "error": str(e)}


async def main():
    """Main demo function for Topic Query Integration"""
    try:
        # Initialize services
        config = get_config()
        config.validate()

        llm_provider = OpenAILLMProvider(config=config)
        llm_service = LLMService(provider=llm_provider)
        query_processor = QueryProcessor(llm_service=llm_service)
        topic_service = TopicQueryService(query_processor=query_processor)

        logger.info("🎯 Topic Query Integration Demo")
        logger.info("=" * 40)

        # 1. Basic topic query processing
        logger.info("📝 Basic Topic Query Processing")
        logger.info("-" * 35)

        test_cases = [
            ("How do I learn programming?", "python"),
            ("What are the best practices?", "web development"),
            ("How to get started?", "machine learning"),
            ("I want to understand the basics", "database design"),
            ("Can you explain the concepts?", "react framework"),
        ]

        for i, (query, topic) in enumerate(test_cases, 1):
            result = await topic_service.process_topic_query(query, topic)

            if result["status"] == "completed":
                logger.info(f"{i}. Original: '{result['original_query']}'")
                logger.info(f"   Topic: '{result['topic_context']}'")
                logger.info(f"   Enhanced: '{result['enhanced_query']}'")
                logger.info(f"   Contextual: '{result['contextual_query']}'")
                logger.info(
                    f"   Time: {result['processing_time_ms']:.1f}ms | Tokens: {result['tokens_used']}"
                )

                if result.get("context_enhancement_failed"):
                    logger.warning(
                        "   ⚠️  Context enhancement failed, using basic enhancement"
                    )
            else:
                logger.error(f"{i}. Failed: {result.get('error', 'Unknown error')}")
            logger.info("")

        # 2. Batch topic processing
        logger.info("📚 Batch Topic Query Processing")
        logger.info("-" * 35)

        batch_test_cases = [
            ("How to build APIs?", "REST"),
            ("What is containerization?", "Docker"),
            ("How to manage state?", "Redux"),
            ("What are microservices?", "architecture"),
        ]

        batch_result = await topic_service.batch_process_topic_queries(batch_test_cases)

        if batch_result["status"] == "completed":
            logger.info(f"✅ Processed {batch_result['total_queries']} topic queries")
            logger.info(f"   Successful: {batch_result['successful_queries']}")
            logger.info(f"   Failed: {batch_result['failed_queries']}")
            logger.info(
                f"   Total time: {batch_result['total_processing_time_ms']:.1f}ms"
            )
            logger.info(f"   Total tokens: {batch_result['total_tokens_used']}")
            logger.info("")

            for result in batch_result["results"]:
                if result["status"] == "completed":
                    logger.info(
                        f"   '{result['original_query']}' + '{result['topic_context']}'"
                    )
                    logger.info(f"   → '{result['contextual_query']}'")
                else:
                    logger.error(
                        f"   Failed: '{result['original_query']}' - {result.get('error', 'Unknown error')}"
                    )
        else:
            logger.error(
                f"❌ Batch processing failed: {batch_result.get('error', 'Unknown error')}"
            )

        # 3. Real-world scenarios
        logger.info("\n🌟 Real-world Integration Scenarios")
        logger.info("-" * 40)

        scenarios = [
            {
                "name": "Educational Content Search",
                "query": "I want to learn about data structures",
                "context": "computer science tutorial",
            },
            {
                "name": "Technical Problem Solving",
                "query": "How to fix authentication issues?",
                "context": "web security",
            },
            {
                "name": "Framework Learning",
                "query": "What are the main concepts?",
                "context": "Vue.js framework",
            },
            {
                "name": "Industry Best Practices",
                "query": "How to optimize performance?",
                "context": "database queries",
            },
        ]

        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"{i}. {scenario['name']}")

            result = await topic_service.process_topic_query(
                scenario["query"], scenario["context"]
            )

            if result["status"] == "completed":
                logger.info(f"   Query: '{result['original_query']}'")
                logger.info(f"   Context: '{result['topic_context']}'")
                logger.info(f"   Result: '{result['contextual_query']}'")
                logger.info(
                    f"   Performance: {result['processing_time_ms']:.1f}ms, {result['tokens_used']} tokens"
                )
            else:
                logger.error(f"   Failed: {result.get('error', 'Unknown error')}")
            logger.info("")

        # 4. Performance comparison
        logger.info("⚡ Performance Comparison")
        logger.info("-" * 25)

        # Compare with and without context
        test_query = "How do I implement authentication?"

        # Without context
        basic_result = await topic_service.process_topic_query(test_query)

        # With context
        contextual_result = await topic_service.process_topic_query(
            test_query, "web security"
        )

        if (
            basic_result["status"] == "completed"
            and contextual_result["status"] == "completed"
        ):
            logger.info("Without context:")
            logger.info(f"   Query: '{basic_result['enhanced_query']}'")
            logger.info(f"   Time: {basic_result['processing_time_ms']:.1f}ms")
            logger.info(f"   Tokens: {basic_result['tokens_used']}")

            logger.info("With context:")
            logger.info(f"   Query: '{contextual_result['contextual_query']}'")
            logger.info(f"   Time: {contextual_result['processing_time_ms']:.1f}ms")
            logger.info(f"   Tokens: {contextual_result['tokens_used']}")

            time_overhead = (
                contextual_result["processing_time_ms"]
                - basic_result["processing_time_ms"]
            )
            token_overhead = (
                contextual_result["tokens_used"] - basic_result["tokens_used"]
            )

            logger.info(
                f"Context overhead: +{time_overhead:.1f}ms, +{token_overhead} tokens"
            )

        logger.info("\n🎉 Topic Query Integration Demo completed successfully!")
        logger.info("=" * 65)
        logger.info("✅ QueryProcessor is ready for topic-based workflows!")
        logger.info("🔧 Use this pattern to integrate with your topic application")

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
