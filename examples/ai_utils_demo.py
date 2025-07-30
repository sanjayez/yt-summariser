#!/usr/bin/env python3
"""
AI Utils Demo Script
Demonstrates vector store and LLM interaction capabilities
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import ai_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_utils.config import get_config
from ai_utils.providers import OpenAIEmbeddingProvider, WeaviateVectorStoreProvider, OpenAILLMProvider
from ai_utils.services import EmbeddingService, VectorService, LLMService
from ai_utils.models import (
    VectorDocument, VectorQuery, RAGQuery, 
    ChatMessage, ChatRole, TextGenerationRequest
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main demo function"""
    try:
        # Get configuration
        config = get_config()
        config.validate()
        
        # Initialize providers
        embedding_provider = OpenAIEmbeddingProvider(config=config)
        vector_provider = WeaviateVectorStoreProvider(config=config)
        llm_provider = OpenAILLMProvider(config=config)
        
        # Initialize services
        embedding_service = EmbeddingService(provider=embedding_provider)
        vector_service = VectorService(provider=vector_provider)
        llm_service = LLMService(provider=llm_provider)
        
        logger.info("🚀 AI Utils Demo - Vector Store + LLM Integration")
        logger.info("=" * 60)
        
        # 1. Health checks
        logger.info("🏥 Running health checks...")
        
        embedding_health = await embedding_service.health_check()
        vector_health = await vector_service.health_check()
        llm_health = await llm_service.health_check()
        
        logger.info(f"Embedding service: {embedding_health['status']}")
        logger.info(f"Vector service: {vector_health['status']}")
        logger.info(f"LLM service: {llm_health['status']}")
        
        if not all(h['status'] == 'healthy' for h in [embedding_health, vector_health, llm_health]):
            logger.error("❌ One or more services are unhealthy. Please check your configuration.")
            return
        
        # 2. Simple LLM text generation
        logger.info("\n📝 Simple Text Generation Demo")
        logger.info("-" * 30)
        
        simple_prompt = "What is machine learning in simple terms?"
        text_result = await llm_service.generate_text(
            prompt=simple_prompt,
            temperature=0.7,
            max_tokens=150
        )
        
        logger.info(f"Prompt: {simple_prompt}")
        logger.info(f"Response: {text_result['text']}")
        logger.info(f"Processing time: {text_result['processing_time_ms']:.2f}ms")
        
        # 3. Chat completion demo
        logger.info("\n💬 Chat Completion Demo")
        logger.info("-" * 25)
        
        chat_request = llm_service.create_simple_chat_request(
            user_message="Explain the difference between supervised and unsupervised learning",
            system_message="You are a helpful AI tutor. Explain concepts clearly and concisely.",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=200
        )
        
        chat_result = await llm_service.chat_completion(chat_request)
        
        logger.info(f"Chat response: {chat_result['response'].choices[0].message.content}")
        logger.info(f"Tokens used: {chat_result['response'].usage.total_tokens}")
        logger.info(f"Processing time: {chat_result['processing_time_ms']:.2f}ms")
        
        # 4. Vector store operations
        logger.info("\n🔍 Vector Store Operations Demo")
        logger.info("-" * 35)
        
        # Sample AI/ML documents
        sample_documents = [
            {
                "id": "ml_basics",
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "metadata": {"category": "machine_learning", "level": "beginner"}
            },
            {
                "id": "deep_learning",
                "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, enabling breakthroughs in computer vision and natural language processing.",
                "metadata": {"category": "deep_learning", "level": "intermediate"}
            },
            {
                "id": "supervised_learning",
                "text": "Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new, unseen data.",
                "metadata": {"category": "machine_learning", "level": "intermediate"}
            },
            {
                "id": "unsupervised_learning",
                "text": "Unsupervised learning finds hidden patterns in data without labeled examples, using techniques like clustering and dimensionality reduction.",
                "metadata": {"category": "machine_learning", "level": "intermediate"}
            },
            {
                "id": "neural_networks",
                "text": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections.",
                "metadata": {"category": "deep_learning", "level": "advanced"}
            }
        ]
        
        # Bulk upsert documents with embeddings
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]
        
        bulk_result = await vector_service.bulk_upsert_with_embedding(
            texts=texts,
            embedding_service=embedding_service,
            metadata_list=metadata_list
        )
        
        logger.info(f"✅ Bulk upserted {bulk_result['upserted_count']} documents in {bulk_result['processing_time_ms']:.2f}ms")
        
        # 5. Vector search demo
        logger.info("\n🔎 Vector Search Demo")
        logger.info("-" * 25)
        
        search_query = "What is neural network architecture?"
        
        # First, get embedding for the query
        embedding_response = await embedding_service.embed_text(search_query)
        
        # Search for similar documents
        query = VectorQuery(
            query=search_query,
            embedding=embedding_response.embedding,
            top_k=3,
            include_metadata=True
        )
        
        search_result = await vector_service.search_similar(query)
        
        logger.info(f"Search query: {search_query}")
        logger.info(f"Found {len(search_result['results'])} results in {search_result['processing_time_ms']:.2f}ms:")
        
        for i, result in enumerate(search_result['results'], 1):
            logger.info(f"  {i}. {result.text[:100]}... (score: {result.score:.3f})")
        
        # 6. RAG Demo - The main integration!
        logger.info("\n🎯 RAG (Retrieval-Augmented Generation) Demo")
        logger.info("-" * 45)
        
        rag_query = RAGQuery(
            query="How do neural networks learn and what makes them different from traditional programming?",
            top_k=3,
            temperature=0.7,
            max_tokens=300
        )
        
        # Get context documents from vector search
        context_embedding = await embedding_service.embed_text(rag_query.query)
        context_query = VectorQuery(
            query=rag_query.query,
            embedding=context_embedding.embedding,
            top_k=rag_query.top_k,
            include_metadata=True
        )
        
        context_search = await vector_service.search_similar(context_query)
        context_documents = context_search['results']
        
        # Generate RAG response
        rag_result = await llm_service.generate_rag_response(
            query=rag_query,
            context_documents=context_documents
        )
        
        logger.info(f"RAG Query: {rag_query.query}")
        logger.info(f"Context documents used: {len(context_documents)}")
        logger.info(f"RAG Answer: {rag_result['response'].answer}")
        logger.info(f"Processing time: {rag_result['processing_time_ms']:.2f}ms")
        logger.info(f"Tokens used: {rag_result['response'].usage['total_tokens']}")
        
        # 7. Batch LLM operations
        logger.info("\n📚 Batch LLM Operations Demo")
        logger.info("-" * 30)
        
        batch_prompts = [
            "What is the main benefit of deep learning?",
            "How does unsupervised learning work?",
            "What are the key components of a neural network?"
        ]
        
        batch_result = await llm_service.batch_text_generation(
            prompts=batch_prompts,
            temperature=0.7,
            max_tokens=100,
            system_prompt="You are a concise AI educator. Provide brief, clear explanations."
        )
        
        logger.info(f"✅ Batch generated {batch_result['total_processed']} responses in {batch_result['processing_time_ms']:.2f}ms")
        
        for i, result in enumerate(batch_result['results'], 1):
            if 'text' in result:
                logger.info(f"  {i}. {result['text']}")
        
        # 8. Performance statistics
        logger.info("\n📊 Performance Statistics")
        logger.info("-" * 25)
        
        embedding_stats = embedding_service.get_performance_stats()
        vector_stats = vector_service.get_performance_stats()
        llm_stats = llm_service.get_performance_stats()
        
        logger.info(f"Embedding service jobs: {embedding_stats['total_jobs']} (completed: {embedding_stats['completed_jobs']})")
        logger.info(f"Vector service jobs: {vector_stats['total_jobs']} (completed: {vector_stats['completed_jobs']})")
        logger.info(f"LLM service jobs: {llm_stats['total_jobs']} (completed: {llm_stats['completed_jobs']})")
        
        # 9. Advanced RAG with filtered search
        logger.info("\n🎓 Advanced RAG with Filtering Demo")
        logger.info("-" * 35)
        
        advanced_rag_query = RAGQuery(
            query="What are the key differences between beginner and advanced AI concepts?",
            top_k=5,
            filters={"level": "intermediate"},  # Only get intermediate level content
            temperature=0.8,
            max_tokens=250
        )
        
        # Get filtered context
        filtered_embedding = await embedding_service.embed_text(advanced_rag_query.query)
        filtered_query = VectorQuery(
            query=advanced_rag_query.query,
            embedding=filtered_embedding.embedding,
            top_k=advanced_rag_query.top_k,
            filters=advanced_rag_query.filters,
            include_metadata=True
        )
        
        filtered_search = await vector_service.search_similar(filtered_query)
        filtered_context = filtered_search['results']
        
        # Generate filtered RAG response
        filtered_rag_result = await llm_service.generate_rag_response(
            query=advanced_rag_query,
            context_documents=filtered_context
        )
        
        logger.info(f"Filtered RAG Query: {advanced_rag_query.query}")
        logger.info(f"Filter applied: {advanced_rag_query.filters}")
        logger.info(f"Filtered context documents: {len(filtered_context)}")
        logger.info(f"Filtered RAG Answer: {filtered_rag_result['response'].answer}")
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 