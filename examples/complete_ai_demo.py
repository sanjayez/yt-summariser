#!/usr/bin/env python3
"""
Complete AI Demo Script - Full Pipeline Demonstration
Shows how vector store and LLM capabilities work together in realistic scenarios
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
)
from ai_utils.providers import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
    WeaviateVectorStoreProvider,
)
from ai_utils.services import EmbeddingService, LLMService, VectorService

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AIKnowledgeBase:
    """Complete AI-powered knowledge base demonstration"""

    def __init__(self):
        self.config = None
        self.embedding_service = None
        self.vector_service = None
        self.llm_service = None
        self.conversation_history = []

    async def initialize(self):
        """Initialize all AI services"""
        logger.info("🚀 Initializing AI Knowledge Base...")

        # Get configuration
        self.config = get_config()
        self.config.validate()

        # Initialize providers
        embedding_provider = OpenAIEmbeddingProvider(config=self.config)
        vector_provider = WeaviateVectorStoreProvider(config=self.config)
        llm_provider = OpenAILLMProvider(config=self.config)

        # Initialize services
        self.embedding_service = EmbeddingService(provider=embedding_provider)
        self.vector_service = VectorService(provider=vector_provider)
        self.llm_service = LLMService(provider=llm_provider)

        # Health checks
        health_results = await asyncio.gather(
            self.embedding_service.health_check(),
            self.vector_service.health_check(),
            self.llm_service.health_check(),
            return_exceptions=True,
        )

        services = ["Embedding", "Vector Store", "LLM"]
        all_healthy = True

        for _i, (service, health) in enumerate(
            zip(services, health_results, strict=False)
        ):
            if isinstance(health, Exception):
                logger.error(f"❌ {service} service failed: {health}")
                all_healthy = False
            elif isinstance(health, dict) and health.get("status") == "healthy":
                logger.info(f"✅ {service} service: healthy")
            else:
                logger.warning(f"⚠️ {service} service: {health}")
                all_healthy = False

        if not all_healthy:
            raise Exception(
                "Some services are not healthy. Please check your configuration."
            )

        logger.info("🎯 All services initialized successfully!")
        return True

    async def populate_knowledge_base(self):
        """Populate the knowledge base with sample documents"""
        logger.info("\n📚 Populating Knowledge Base...")

        # Sample knowledge base documents about AI/ML topics
        documents = [
            {
                "text": "Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.",
                "metadata": {
                    "category": "machine_learning",
                    "difficulty": "beginner",
                    "topic": "fundamentals",
                },
            },
            {
                "text": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that adjusts as learning proceeds, allowing the network to recognize patterns and make predictions.",
                "metadata": {
                    "category": "neural_networks",
                    "difficulty": "intermediate",
                    "topic": "architecture",
                },
            },
            {
                "text": "Deep Learning is a subset of machine learning that uses neural networks with multiple hidden layers to model and understand complex patterns in data. It has enabled breakthroughs in computer vision, natural language processing, and speech recognition.",
                "metadata": {
                    "category": "deep_learning",
                    "difficulty": "intermediate",
                    "topic": "advanced_ml",
                },
            },
            {
                "text": "Supervised Learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new, unseen data. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values).",
                "metadata": {
                    "category": "machine_learning",
                    "difficulty": "beginner",
                    "topic": "learning_types",
                },
            },
            {
                "text": "Unsupervised Learning finds hidden patterns in data without labeled examples. It includes techniques like clustering (grouping similar data points), dimensionality reduction (simplifying data while preserving important features), and association rule mining.",
                "metadata": {
                    "category": "machine_learning",
                    "difficulty": "intermediate",
                    "topic": "learning_types",
                },
            },
            {
                "text": "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative reward over time.",
                "metadata": {
                    "category": "machine_learning",
                    "difficulty": "advanced",
                    "topic": "learning_types",
                },
            },
            {
                "text": "Overfitting occurs when a machine learning model learns the training data too well, including noise and irrelevant patterns. This leads to poor performance on new data. Common prevention techniques include cross-validation, regularization, and early stopping.",
                "metadata": {
                    "category": "machine_learning",
                    "difficulty": "intermediate",
                    "topic": "model_evaluation",
                },
            },
            {
                "text": "Feature Engineering is the process of selecting, modifying, or creating new features from raw data to improve machine learning model performance. Good features can make the difference between a mediocre and excellent model.",
                "metadata": {
                    "category": "machine_learning",
                    "difficulty": "intermediate",
                    "topic": "data_preprocessing",
                },
            },
            {
                "text": "Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers that apply filters to detect features like edges, textures, and patterns in images.",
                "metadata": {
                    "category": "deep_learning",
                    "difficulty": "advanced",
                    "topic": "computer_vision",
                },
            },
            {
                "text": "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language. Modern NLP uses transformer models like BERT and GPT for tasks like translation, sentiment analysis, and text generation.",
                "metadata": {
                    "category": "nlp",
                    "difficulty": "advanced",
                    "topic": "language_processing",
                },
            },
        ]

        # Extract texts and metadata for bulk upsert
        texts = [doc["text"] for doc in documents]
        metadata_list = [doc["metadata"] for doc in documents]

        # Bulk upsert with automatic embedding
        try:
            result = await self.vector_service.bulk_upsert_with_embedding(
                texts=texts,
                embedding_service=self.embedding_service,
                metadata_list=metadata_list,
            )

            logger.info(
                f"✅ Successfully stored {result['upserted_count']} documents in {result.get('processing_time_ms', 0):.1f}ms"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to populate knowledge base: {e}")
            # For demo purposes, continue even if vector store fails
            logger.info("📝 Continuing with LLM-only demonstrations...")
            return False

    async def demonstrate_search_capabilities(self):
        """Demonstrate vector search capabilities"""
        logger.info("\n🔍 Demonstrating Search Capabilities...")

        search_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What is the difference between supervised and unsupervised learning?",
            "How to prevent overfitting in models?",
        ]

        for i, query in enumerate(search_queries, 1):
            try:
                logger.info(f"\n{i}. Searching: '{query}'")

                # Search for relevant documents
                search_result = await self.vector_service.search_by_text(
                    text=query,
                    embedding_service=self.embedding_service,
                    top_k=3,
                    include_metadata=True,
                )

                logger.info(f"   Found {len(search_result.results)} results:")
                for j, result in enumerate(search_result.results, 1):
                    logger.info(
                        f"   {j}. Score: {result.score:.3f} | Category: {result.metadata.get('category', 'N/A')}"
                    )
                    logger.info(f"      Text: {result.text[:100]}...")

            except Exception as e:
                logger.error(f"   ❌ Search failed: {e}")

    async def demonstrate_llm_capabilities(self):
        """Demonstrate LLM capabilities"""
        logger.info("\n🤖 Demonstrating LLM Capabilities...")

        # 1. Simple text generation
        logger.info("\n1. Simple Text Generation:")
        try:
            result = await self.llm_service.generate_text(
                prompt="Explain artificial intelligence in simple terms for a beginner",
                temperature=0.7,
                max_tokens=150,
            )
            logger.info(f"   Generated: {result['text']}")
            logger.info(f"   Time: {result['processing_time_ms']:.1f}ms")
        except Exception as e:
            logger.error(f"   ❌ Text generation failed: {e}")

        # 2. Chat with system prompt
        logger.info("\n2. Chat Completion with System Prompt:")
        try:
            chat_request = self.llm_service.create_simple_chat_request(
                user_message="What are the main types of machine learning?",
                system_message="You are an AI tutor. Provide clear, structured explanations with examples.",
                temperature=0.6,
                max_tokens=200,
            )

            chat_result = await self.llm_service.chat_completion(chat_request)
            response_text = chat_result["response"].choices[0].message.content

            logger.info(f"   Response: {response_text}")
            logger.info(f"   Tokens: {chat_result['response'].usage.total_tokens}")

        except Exception as e:
            logger.error(f"   ❌ Chat completion failed: {e}")

        # 3. Batch text generation
        logger.info("\n3. Batch Text Generation:")
        try:
            prompts = [
                "Define machine learning in one sentence",
                "Define deep learning in one sentence",
                "Define artificial intelligence in one sentence",
            ]

            batch_result = await self.llm_service.batch_text_generation(
                prompts=prompts,
                temperature=0.5,
                max_tokens=50,
                system_prompt="Provide concise, accurate definitions.",
            )

            logger.info(
                f"   Generated {batch_result['total_processed']} responses in {batch_result['processing_time_ms']:.1f}ms:"
            )
            for i, result in enumerate(batch_result["results"], 1):
                if "text" in result:
                    logger.info(f"   {i}. {result['text']}")

        except Exception as e:
            logger.error(f"   ❌ Batch generation failed: {e}")

    async def demonstrate_rag_pipeline(self):
        """Demonstrate RAG (Retrieval-Augmented Generation) pipeline"""
        logger.info("\n🎯 Demonstrating RAG Pipeline...")

        rag_questions = [
            "How do neural networks learn and what makes them different from traditional programming?",
            "What's the difference between supervised and unsupervised learning, and when should I use each?",
            "What is overfitting and how can I prevent it in my machine learning models?",
        ]

        for i, question in enumerate(rag_questions, 1):
            logger.info(f"\n{i}. RAG Question: '{question}'")

            try:
                # Step 1: Search for relevant context
                search_result = await self.vector_service.search_by_text(
                    text=question,
                    embedding_service=self.embedding_service,
                    top_k=3,
                    include_metadata=True,
                )

                logger.info(
                    f"   📚 Found {len(search_result.results)} context documents"
                )

                # Step 2: Generate RAG response
                rag_query = RAGQuery(
                    query=question, top_k=3, temperature=0.7, max_tokens=250
                )

                rag_result = await self.llm_service.generate_rag_response(
                    query=rag_query, context_documents=search_result.results
                )

                logger.info(f"   🤖 RAG Answer: {rag_result['response'].answer}")
                logger.info(
                    f"   📊 Processing time: {rag_result['processing_time_ms']:.1f}ms"
                )
                logger.info(
                    f"   🔢 Tokens used: {rag_result['response'].usage['total_tokens']}"
                )

                # Show which sources were used
                logger.info("   📖 Sources used:")
                for j, source in enumerate(rag_result["response"].sources, 1):
                    category = source.metadata.get("category", "N/A")
                    logger.info(f"      {j}. {category} (score: {source.score:.3f})")

            except Exception as e:
                logger.error(f"   ❌ RAG failed: {e}")

                # Fallback to LLM-only response
                try:
                    logger.info("   🔄 Falling back to LLM-only response...")
                    fallback_result = await self.llm_service.generate_text(
                        prompt=question, temperature=0.7, max_tokens=200
                    )
                    logger.info(f"   🤖 Fallback Answer: {fallback_result['text']}")
                except Exception as fallback_error:
                    logger.error(f"   ❌ Fallback also failed: {fallback_error}")

    async def demonstrate_conversation(self):
        """Demonstrate multi-turn conversation"""
        logger.info("\n💬 Demonstrating Multi-turn Conversation...")

        # Initialize conversation with system prompt
        self.conversation_history = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a helpful AI assistant specializing in machine learning education. Provide clear, accurate explanations and ask follow-up questions when appropriate.",
            )
        ]

        # Conversation turns
        user_messages = [
            "I'm new to machine learning. Where should I start?",
            "What's the difference between classification and regression?",
            "Can you give me a simple example of each?",
            "What programming language should I learn first?",
        ]

        for i, user_message in enumerate(user_messages, 1):
            logger.info(f"\n{i}. User: {user_message}")

            try:
                # Add user message to conversation
                self.conversation_history.append(
                    ChatMessage(role=ChatRole.USER, content=user_message)
                )

                # Get response from LLM
                from ai_utils.models import ChatRequest

                chat_request = ChatRequest(
                    messages=self.conversation_history,
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=200,
                )

                chat_result = await self.llm_service.chat_completion(chat_request)
                assistant_response = chat_result["response"].choices[0].message.content

                # Add assistant response to conversation
                self.conversation_history.append(
                    ChatMessage(role=ChatRole.ASSISTANT, content=assistant_response)
                )

                logger.info(f"   Assistant: {assistant_response}")
                logger.info(f"   Tokens: {chat_result['response'].usage.total_tokens}")

            except Exception as e:
                logger.error(f"   ❌ Conversation turn failed: {e}")

    async def demonstrate_intelligent_search(self):
        """Demonstrate intelligent search with LLM enhancement"""
        logger.info("\n🧠 Demonstrating Intelligent Search...")

        search_query = (
            "I want to build a system that can recognize images. What should I learn?"
        )

        try:
            logger.info(f"Query: '{search_query}'")

            # Step 1: Search for relevant documents
            search_result = await self.vector_service.search_by_text(
                text=search_query,
                embedding_service=self.embedding_service,
                top_k=5,
                include_metadata=True,
            )

            # Step 2: Use LLM to analyze and enhance the search results
            context_summary = "\n".join(
                [
                    f"- {result.text[:100]}... (Category: {result.metadata.get('category', 'N/A')})"
                    for result in search_result.results
                ]
            )

            enhancement_prompt = f"""Based on this search query: "{search_query}"

And these relevant knowledge base results:
{context_summary}

Provide a comprehensive learning path and recommendations. Structure your response with:
1. Key concepts to understand
2. Recommended learning order
3. Practical next steps
4. Additional resources to explore"""

            enhanced_result = await self.llm_service.generate_text(
                prompt=enhancement_prompt,
                temperature=0.7,
                max_tokens=300,
                system_prompt="You are an AI education advisor. Provide structured, actionable learning guidance.",
            )

            logger.info("🎯 Intelligent Search Result:")
            logger.info(f"{enhanced_result['text']}")
            logger.info(
                f"📊 Based on {len(search_result.results)} knowledge base documents"
            )

        except Exception as e:
            logger.error(f"❌ Intelligent search failed: {e}")

    async def show_performance_statistics(self):
        """Show performance statistics for all services"""
        logger.info("\n📊 Performance Statistics...")

        try:
            # Get stats from all services
            embedding_stats = self.embedding_service.get_performance_stats()
            vector_stats = self.vector_service.get_performance_stats()
            llm_stats = self.llm_service.get_performance_stats()

            logger.info("📈 Embedding Service:")
            logger.info(f"   Total jobs: {embedding_stats['total_jobs']}")
            logger.info(f"   Completed: {embedding_stats['completed_jobs']}")
            logger.info(f"   Failed: {embedding_stats['failed_jobs']}")

            logger.info("🔍 Vector Service:")
            logger.info(f"   Total jobs: {vector_stats['total_jobs']}")
            logger.info(f"   Completed: {vector_stats['completed_jobs']}")
            logger.info(f"   Failed: {vector_stats['failed_jobs']}")

            logger.info("🤖 LLM Service:")
            logger.info(f"   Total jobs: {llm_stats['total_jobs']}")
            logger.info(f"   Completed: {llm_stats['completed_jobs']}")
            logger.info(f"   Failed: {llm_stats['failed_jobs']}")

            # Show some performance metrics if available
            if llm_stats.get("benchmark_stats"):
                logger.info("⚡ Performance Metrics:")
                for operation, metrics in llm_stats["benchmark_stats"].items():
                    if "avg_ms" in metrics:
                        logger.info(
                            f"   {operation}: avg {metrics['avg_ms']:.1f}ms ({metrics['count']} calls)"
                        )

        except Exception as e:
            logger.error(f"❌ Failed to get performance statistics: {e}")


async def main():
    """Main demonstration function"""
    try:
        logger.info("🌟 Complete AI Demo - Vector Store + LLM Integration")
        logger.info("=" * 70)

        # Initialize the AI knowledge base
        kb = AIKnowledgeBase()
        await kb.initialize()

        # Run all demonstrations
        vector_store_available = await kb.populate_knowledge_base()

        if vector_store_available:
            await kb.demonstrate_search_capabilities()
            await kb.demonstrate_rag_pipeline()
            await kb.demonstrate_intelligent_search()

        await kb.demonstrate_llm_capabilities()
        await kb.demonstrate_conversation()
        await kb.show_performance_statistics()

        logger.info("\n🎉 Complete AI Demo finished successfully!")
        logger.info("=" * 70)
        logger.info("✅ You now have a complete AI infrastructure with:")
        logger.info("   📚 Vector Store - Document embedding and semantic search")
        logger.info("   🤖 LLM Integration - Text generation and chat completions")
        logger.info("   🎯 RAG Pipeline - Retrieval-Augmented Generation")
        logger.info("   💬 Conversational AI - Multi-turn conversations")
        logger.info("   🧠 Intelligent Search - LLM-enhanced search results")
        logger.info("   📊 Performance Monitoring - Job tracking and metrics")
        logger.info("\n🚀 Ready to build advanced AI applications!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        logger.info("\n🔧 Troubleshooting tips:")
        logger.info("   1. Check your OpenAI API key is set: OPENAI_API_KEY")
        logger.info("   2. Verify Weaviate configuration (API key, region, index)")
        logger.info("   3. Ensure you have internet connectivity")
        logger.info("   4. Check the logs above for specific error details")
        raise


if __name__ == "__main__":
    asyncio.run(main())
