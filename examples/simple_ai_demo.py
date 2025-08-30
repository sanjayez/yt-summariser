#!/usr/bin/env python3
"""
Simple AI Demo Script - Pure LLM Capabilities
Demonstrates LLM functionality without vector store dependencies
Perfect for testing and understanding the LLM layer independently
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import ai_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_utils.config import get_config
from ai_utils.models import ChatMessage, ChatRole
from ai_utils.providers import OpenAILLMProvider
from ai_utils.services import LLMService

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_text_generation(llm_service):
    """Demonstrate simple text generation"""
    logger.info("\n📝 Text Generation Demo")
    logger.info("-" * 30)

    prompts = [
        "Explain machine learning in one paragraph",
        "What are the benefits of using Python for data science?",
        "Describe the difference between AI and machine learning",
    ]

    for i, prompt in enumerate(prompts, 1):
        try:
            result = await llm_service.generate_text(
                prompt=prompt, temperature=0.7, max_tokens=120
            )

            logger.info(f"{i}. Prompt: {prompt}")
            logger.info(f"   Response: {result['text']}")
            logger.info(f"   Time: {result['processing_time_ms']:.1f}ms\n")

        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")


async def demo_chat_completion(llm_service):
    """Demonstrate chat completion with system prompts"""
    logger.info("💬 Chat Completion Demo")
    logger.info("-" * 25)

    scenarios = [
        {
            "system": "You are a helpful programming tutor",
            "user": "What's the best way to learn Python programming?",
            "context": "Programming Education",
        },
        {
            "system": "You are a technical interviewer",
            "user": "Explain what a REST API is",
            "context": "Technical Interview",
        },
        {
            "system": "You are a data science mentor",
            "user": "How do I choose the right machine learning algorithm?",
            "context": "Data Science Mentoring",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        try:
            chat_request = llm_service.create_simple_chat_request(
                user_message=scenario["user"],
                system_message=scenario["system"],
                temperature=0.6,
                max_tokens=150,
            )

            result = await llm_service.chat_completion(chat_request)
            response = result["response"].choices[0].message.content

            logger.info(f"{i}. Context: {scenario['context']}")
            logger.info(f"   User: {scenario['user']}")
            logger.info(f"   Assistant: {response}")
            logger.info(f"   Tokens: {result['response'].usage.total_tokens}")
            logger.info(f"   Time: {result['processing_time_ms']:.1f}ms\n")

        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")


async def demo_conversation_flow(llm_service):
    """Demonstrate multi-turn conversation"""
    logger.info("🗣️ Conversation Flow Demo")
    logger.info("-" * 30)

    # Start conversation
    conversation = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="You are a helpful AI assistant that provides clear, concise answers about technology.",
        )
    ]

    user_inputs = [
        "What is cloud computing?",
        "What are the main benefits?",
        "Which cloud provider would you recommend for a startup?",
        "What should I consider when choosing?",
    ]

    for i, user_input in enumerate(user_inputs, 1):
        try:
            # Add user message
            conversation.append(ChatMessage(role=ChatRole.USER, content=user_input))

            # Get response
            from ai_utils.models import ChatRequest

            chat_request = ChatRequest(
                messages=conversation,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100,
            )

            result = await llm_service.chat_completion(chat_request)
            assistant_response = result["response"].choices[0].message.content

            # Add assistant response to conversation
            conversation.append(
                ChatMessage(role=ChatRole.ASSISTANT, content=assistant_response)
            )

            logger.info(f"{i}. User: {user_input}")
            logger.info(f"   Assistant: {assistant_response}")
            logger.info(f"   Tokens: {result['response'].usage.total_tokens}\n")

        except Exception as e:
            logger.error(f"   ❌ Turn {i} failed: {e}")


async def demo_batch_processing(llm_service):
    """Demonstrate batch text generation"""
    logger.info("📚 Batch Processing Demo")
    logger.info("-" * 30)

    # Create a batch of related prompts
    prompts = [
        "Define artificial intelligence",
        "Define machine learning",
        "Define deep learning",
        "Define neural networks",
        "Define natural language processing",
    ]

    try:
        result = await llm_service.batch_text_generation(
            prompts=prompts,
            temperature=0.5,
            max_tokens=60,
            system_prompt="Provide clear, technical definitions in 1-2 sentences.",
        )

        logger.info(
            f"✅ Generated {result['total_processed']} definitions in {result['processing_time_ms']:.1f}ms:"
        )

        for i, response in enumerate(result["results"], 1):
            if "text" in response:
                logger.info(f"{i}. {prompts[i - 1]}")
                logger.info(f"   {response['text']}\n")

    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")


async def demo_creative_tasks(llm_service):
    """Demonstrate creative and analytical tasks"""
    logger.info("🎨 Creative & Analytical Tasks Demo")
    logger.info("-" * 40)

    tasks = [
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate the Fibonacci sequence",
            "system": "You are a helpful programming assistant. Write clean, well-commented code.",
        },
        {
            "name": "Data Analysis",
            "prompt": "How would you analyze customer churn data? List the key steps.",
            "system": "You are a data science consultant. Provide practical, actionable advice.",
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a short story about an AI that learns to paint",
            "system": "You are a creative writing assistant. Write engaging, imaginative content.",
        },
        {
            "name": "Problem Solving",
            "prompt": "A company's website is loading slowly. What could be the causes and solutions?",
            "system": "You are a technical consultant. Provide systematic problem-solving approaches.",
        },
    ]

    for i, task in enumerate(tasks, 1):
        try:
            chat_request = llm_service.create_simple_chat_request(
                user_message=task["prompt"],
                system_message=task["system"],
                temperature=0.7,
                max_tokens=200,
            )

            result = await llm_service.chat_completion(chat_request)
            response = result["response"].choices[0].message.content

            logger.info(f"{i}. Task: {task['name']}")
            logger.info(f"   Prompt: {task['prompt']}")
            logger.info(f"   Response: {response}")
            logger.info(f"   Time: {result['processing_time_ms']:.1f}ms\n")

        except Exception as e:
            logger.error(f"   ❌ Task {task['name']} failed: {e}")


async def show_performance_summary(llm_service):
    """Show performance summary"""
    logger.info("📊 Performance Summary")
    logger.info("-" * 25)

    try:
        stats = llm_service.get_performance_stats()

        logger.info(f"Total LLM operations: {stats['total_jobs']}")
        logger.info(f"Successful operations: {stats['completed_jobs']}")
        logger.info(f"Failed operations: {stats['failed_jobs']}")

        if stats.get("benchmark_stats"):
            logger.info("Performance metrics:")
            for operation, metrics in stats["benchmark_stats"].items():
                if "avg_ms" in metrics:
                    logger.info(
                        f"  {operation}: avg {metrics['avg_ms']:.1f}ms ({metrics['count']} calls)"
                    )

    except Exception as e:
        logger.error(f"❌ Failed to get performance stats: {e}")


async def main():
    """Main demonstration function"""
    try:
        logger.info("🤖 Simple AI Demo - LLM Capabilities")
        logger.info("=" * 50)

        # Initialize configuration
        config = get_config()
        config.validate()

        # Initialize LLM service only
        llm_provider = OpenAILLMProvider(config=config)
        llm_service = LLMService(provider=llm_provider)

        # Health check
        logger.info("🏥 Running health check...")
        health = await llm_service.health_check()

        if health["status"] != "healthy":
            logger.error(f"❌ LLM service is not healthy: {health}")
            return

        logger.info("✅ LLM service is healthy!")
        logger.info(f"Supported models: {health['supported_models'][:3]}...")

        # Run all demonstrations
        await demo_text_generation(llm_service)
        await demo_chat_completion(llm_service)
        await demo_conversation_flow(llm_service)
        await demo_batch_processing(llm_service)
        await demo_creative_tasks(llm_service)
        await show_performance_summary(llm_service)

        logger.info("\n🎉 Simple AI Demo completed successfully!")
        logger.info("=" * 50)
        logger.info("✅ LLM capabilities demonstrated:")
        logger.info("   📝 Text generation with various prompts")
        logger.info("   💬 Chat completions with system prompts")
        logger.info("   🗣️ Multi-turn conversations")
        logger.info("   📚 Batch processing for efficiency")
        logger.info("   🎨 Creative and analytical tasks")
        logger.info("   📊 Performance monitoring")
        logger.info(
            "\n🚀 Ready to integrate LLM capabilities into your Django project!"
        )

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("   1. Ensure OPENAI_API_KEY is set in your environment")
        logger.info("   2. Check your internet connection")
        logger.info("   3. Verify OpenAI API access and quotas")
        raise


if __name__ == "__main__":
    asyncio.run(main())
