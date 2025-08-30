#!/usr/bin/env python3
"""
LLM Demo Script - AI Utils LLM Interaction Layer
Demonstrates the LLM interaction capabilities independently
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import ai_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_utils.config import get_config
from ai_utils.models import (
    ChatMessage,
    ChatRole,
    RAGQuery,
    TextGenerationRequest,
    VectorSearchResult,
)
from ai_utils.providers import OpenAILLMProvider
from ai_utils.services import LLMService

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main demo function for LLM interactions"""
    try:
        # Get configuration
        config = get_config()
        config.validate()

        # Initialize LLM provider and service
        llm_provider = OpenAILLMProvider(config=config)
        llm_service = LLMService(provider=llm_provider)

        logger.info("🤖 AI Utils LLM Demo - Language Model Interaction Layer")
        logger.info("=" * 65)

        # 1. Health check
        logger.info("🏥 Running LLM health check...")

        llm_health = await llm_service.health_check()
        logger.info(f"LLM service: {llm_health['status']}")

        if llm_health["status"] != "healthy":
            logger.error(
                "❌ LLM service is unhealthy. Please check your OpenAI API configuration."
            )
            return

        logger.info(
            f"Supported models: {llm_health['supported_models'][:3]}..."
        )  # Show first 3

        # 2. Simple Text Generation
        logger.info("\n📝 Simple Text Generation")
        logger.info("-" * 30)

        prompts = [
            "What is artificial intelligence?",
            "Explain neural networks in one paragraph",
            "What are the benefits of machine learning?",
        ]

        for i, prompt in enumerate(prompts, 1):
            result = await llm_service.generate_text(
                prompt=prompt, temperature=0.7, max_tokens=100
            )

            logger.info(f"{i}. Q: {prompt}")
            logger.info(f"   A: {result['text']}")
            logger.info(f"   Time: {result['processing_time_ms']:.1f}ms")
            logger.info("")

        # 3. Chat Completion with System Prompt
        logger.info("💬 Chat Completion with System Prompt")
        logger.info("-" * 40)

        chat_request = llm_service.create_simple_chat_request(
            user_message="Explain the difference between machine learning and deep learning",
            system_message="You are a technical educator. Provide clear, structured explanations with examples.",
            model="gpt-3.5-turbo",
            temperature=0.6,
            max_tokens=250,
        )

        chat_result = await llm_service.chat_completion(chat_request)

        logger.info(f"System: {chat_request.messages[0].content}")
        logger.info(f"User: {chat_request.messages[1].content}")
        logger.info(f"Assistant: {chat_result['response'].choices[0].message.content}")
        logger.info(f"Tokens used: {chat_result['response'].usage.total_tokens}")
        logger.info(f"Processing time: {chat_result['processing_time_ms']:.1f}ms")

        # 4. Detailed Text Generation
        logger.info("\n🔧 Detailed Text Generation with Request/Response")
        logger.info("-" * 50)

        detailed_request = TextGenerationRequest(
            prompt="Write a brief introduction to the concept of artificial general intelligence (AGI)",
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=150,
            system_prompt="You are a science communicator writing for a general audience.",
        )

        detailed_result = await llm_service.generate_text_with_response(
            detailed_request
        )

        logger.info(f"Request: {detailed_request.prompt}")
        logger.info(f"Response: {detailed_result['response'].text}")
        logger.info(f"Model: {detailed_result['response'].model}")
        logger.info(
            f"Usage: {detailed_result['response'].usage.prompt_tokens} prompt + {detailed_result['response'].usage.completion_tokens} completion = {detailed_result['response'].usage.total_tokens} total tokens"
        )
        logger.info(
            f"Generation time: {detailed_result['response'].generation_time_ms:.1f}ms"
        )

        # 5. Batch Text Generation
        logger.info("\n📚 Batch Text Generation")
        logger.info("-" * 30)

        batch_prompts = [
            "Define supervised learning",
            "Define unsupervised learning",
            "Define reinforcement learning",
            "Define transfer learning",
        ]

        batch_result = await llm_service.batch_text_generation(
            prompts=batch_prompts,
            temperature=0.5,
            max_tokens=80,
            system_prompt="Provide concise, technical definitions.",
        )

        logger.info(
            f"✅ Generated {batch_result['total_processed']} responses in {batch_result['processing_time_ms']:.1f}ms"
        )

        for i, result in enumerate(batch_result["results"], 1):
            if "text" in result:
                logger.info(f"{i}. {batch_prompts[i - 1]}")
                logger.info(f"   {result['text']}")

        # 6. RAG Simulation (without actual vector search)
        logger.info("\n🎯 RAG Simulation (with mock context documents)")
        logger.info("-" * 45)

        # Mock context documents (simulating vector search results)
        mock_context_documents = [
            VectorSearchResult(
                id="doc1",
                text="Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
                score=0.85,
                metadata={"source": "ml_textbook", "chapter": "neural_networks"},
            ),
            VectorSearchResult(
                id="doc2",
                text="Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers to model and understand complex patterns in data.",
                score=0.78,
                metadata={"source": "ai_handbook", "section": "deep_learning"},
            ),
            VectorSearchResult(
                id="doc3",
                text="Backpropagation is the standard algorithm for training neural networks. It works by calculating gradients and updating weights to minimize the loss function.",
                score=0.72,
                metadata={"source": "algorithms_guide", "topic": "optimization"},
            ),
        ]

        rag_query = RAGQuery(
            query="How do neural networks learn and what is backpropagation?",
            top_k=3,
            temperature=0.7,
            max_tokens=200,
        )

        rag_result = await llm_service.generate_rag_response(
            query=rag_query, context_documents=mock_context_documents
        )

        logger.info(f"RAG Query: {rag_query.query}")
        logger.info(f"Context docs: {len(mock_context_documents)} documents")
        logger.info(f"RAG Answer: {rag_result['response'].answer}")
        logger.info(f"Sources used: {len(rag_result['response'].sources)}")
        logger.info(f"Processing time: {rag_result['processing_time_ms']:.1f}ms")
        logger.info(f"Tokens used: {rag_result['response'].usage['total_tokens']}")

        # 7. Multi-turn Conversation
        logger.info("\n🗣️  Multi-turn Conversation")
        logger.info("-" * 30)

        conversation_messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a helpful AI assistant discussing machine learning concepts.",
            ),
            ChatMessage(
                role=ChatRole.USER, content="What is overfitting in machine learning?"
            ),
        ]

        # First response
        first_chat = llm_service.create_simple_chat_request(
            user_message="What is overfitting in machine learning?",
            system_message="You are a helpful AI assistant discussing machine learning concepts.",
        )
        first_result = await llm_service.chat_completion(first_chat)
        first_response = first_result["response"].choices[0].message.content

        # Add assistant response to conversation
        conversation_messages.append(
            ChatMessage(role=ChatRole.ASSISTANT, content=first_response)
        )
        conversation_messages.append(
            ChatMessage(role=ChatRole.USER, content="How can I prevent overfitting?")
        )

        # Second response
        from ai_utils.models import ChatRequest

        followup_request = ChatRequest(
            messages=conversation_messages,
            model="gpt-3.5-turbo",
            temperature=0.6,
            max_tokens=150,
        )

        followup_result = await llm_service.chat_completion(followup_request)
        followup_response = followup_result["response"].choices[0].message.content

        logger.info("Conversation:")
        logger.info("User: What is overfitting in machine learning?")
        logger.info(f"Assistant: {first_response}")
        logger.info("User: How can I prevent overfitting?")
        logger.info(f"Assistant: {followup_response}")

        # 8. Performance Statistics
        logger.info("\n📊 Performance Statistics")
        logger.info("-" * 25)

        stats = llm_service.get_performance_stats()

        logger.info(f"Total LLM jobs: {stats['total_jobs']}")
        logger.info(f"Completed jobs: {stats['completed_jobs']}")
        logger.info(f"Failed jobs: {stats['failed_jobs']}")

        # Show some performance metrics
        if stats["benchmark_stats"]:
            for operation, metrics in stats["benchmark_stats"].items():
                logger.info(
                    f"{operation}: avg {metrics['avg_ms']:.1f}ms (count: {metrics['count']})"
                )

        logger.info("\n🎉 LLM Demo completed successfully!")
        logger.info("=" * 65)
        logger.info("✅ The AI Utils LLM interaction layer is fully functional!")
        logger.info(
            "🔧 You can now use both vector store and LLM capabilities together"
        )

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
