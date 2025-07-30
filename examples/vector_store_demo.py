"""
Vector Store Service Demonstration

This script demonstrates the vector store service capabilities including:
- Document CRUD operations
- Similarity search
- Index management
- Performance benchmarking
- Health checks
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI utilities
from ai_utils.providers import WeaviateVectorStoreProvider, OpenAIEmbeddingProvider
from ai_utils.services import VectorService, EmbeddingService
from ai_utils.models import (
    VectorDocument, VectorQuery, VectorSearchResponse,
    EmbeddingResponse, IndexConfig, IndexStats
)
from ai_utils.config import get_config
from ai_utils.utils.performance import PerformanceBenchmark

async def demonstrate_vector_store_operations():
    """Demonstrate vector store operations"""
    
    logger.info("🚀 Starting Vector Store Service Demonstration")
    
    # Initialize services
    config = get_config()
    
    # Initialize embedding service
    embedding_provider = OpenAIEmbeddingProvider(config=config)
    embedding_service = EmbeddingService(provider=embedding_provider)
    
    # Initialize vector service
    vector_provider = WeaviateVectorStoreProvider(config=config)
    vector_service = VectorService(provider=vector_provider)
    
    # Health checks
    logger.info("🔍 Performing health checks...")
    
    embedding_health = await embedding_service.health_check()
    vector_health = await vector_service.health_check()
    
    logger.info(f"Embedding service health: {'✅ Healthy' if embedding_health else '❌ Unhealthy'}")
    logger.info(f"Vector service health: {'✅ Healthy' if vector_health else '❌ Unhealthy'}")
    
    if not embedding_health or not vector_health:
        logger.error("❌ Health checks failed. Please check your configuration.")
        return
    
    # List existing indexes
    logger.info("📋 Listing existing indexes...")
    try:
        indexes = await vector_service.list_indexes()
        logger.info(f"Found {len(indexes)} indexes:")
        for index in indexes:
            logger.info(f"  - {index.name} (dimension: {index.dimension}, metric: {index.metric})")
    except Exception as e:
        logger.warning(f"Could not list indexes: {e}")
    
    # Get index statistics
    logger.info("📊 Getting index statistics...")
    try:
        stats = await vector_service.get_index_stats()
        logger.info(f"Index stats: {stats.total_vector_count} vectors, {stats.dimension} dimensions, {stats.index_fullness:.2%} fullness")
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
    
    # Sample documents for demonstration
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from images.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Data science combines statistics, programming, and domain expertise to extract insights.",
        "Big data refers to large, complex datasets that require specialized tools for analysis.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Blockchain technology creates secure, decentralized ledgers for digital transactions.",
        "Internet of Things connects physical devices to the internet for data collection and control."
    ]
    
    # Create documents with embeddings
    logger.info("📝 Creating sample documents with embeddings...")
    
    documents = []
    for i, text in enumerate(sample_texts):
        # Embed the text
        embedding_response = await embedding_service.embed_text(text)
        
        # Create document
        document = VectorDocument(
            id=f"doc_{i+1}",
            text=text,
            embedding=embedding_response.embedding,
            metadata={
                "category": "ai_technology",
                "length": len(text),
                "topic": "artificial_intelligence" if "AI" in text or "artificial intelligence" in text.lower() else "technology"
            }
        )
        documents.append(document)
    
    # Upsert documents
    logger.info("💾 Upserting documents to vector store...")
    
    with PerformanceBenchmark("Document Upsert") as benchmark:
        result = await vector_service.upsert_documents(documents, job_id="demo_upsert")
    
    logger.info(f"✅ Upserted {result['upserted_count']} documents in {benchmark.elapsed_time:.2f}ms")
    logger.info(f"Job status: {result['status']}")
    
    # Wait a moment for indexing
    logger.info("⏳ Waiting for indexing to complete...")
    await asyncio.sleep(2)
    
    # Search by text
    logger.info("🔍 Performing text-based searches...")
    
    search_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Tell me about natural language processing",
        "What is computer vision?",
        "Explain reinforcement learning"
    ]
    
    for query in search_queries:
        logger.info(f"\n🔎 Searching for: '{query}'")
        
        with PerformanceBenchmark("Text Search") as benchmark:
            response = await vector_service.search_by_text(
                text=query,
                embedding_service=embedding_service,
                top_k=3,
                include_metadata=True
            )
        
        logger.info(f"Found {len(response.results)} results in {benchmark.elapsed_time:.2f}ms:")
        
        for i, result in enumerate(response.results, 1):
            logger.info(f"  {i}. {result.text[:80]}... (score: {result.score:.3f})")
            logger.info(f"     Metadata: {result.metadata}")
    
    # Search with filters
    logger.info("\n🔍 Performing filtered searches...")
    
    # Search for AI-related documents
    embedding_response = await embedding_service.embed_text("artificial intelligence")
    ai_query = VectorQuery(
        query="artificial intelligence",
        embedding=embedding_response.embedding,
        top_k=5,
        filters={"topic": "artificial_intelligence"},
        include_metadata=True
    )
    
    with PerformanceBenchmark("Filtered Search") as benchmark:
        ai_response = await vector_service.search_similar(ai_query)
    
    logger.info(f"Found {len(ai_response.results)} AI-related documents in {benchmark.elapsed_time:.2f}ms:")
    for i, result in enumerate(ai_response.results, 1):
        logger.info(f"  {i}. {result.text[:80]}... (score: {result.score:.3f})")
    
    # Get specific document
    logger.info("\n📄 Retrieving specific document...")
    
    try:
        document = await vector_service.get_document("doc_1")
        if document:
            logger.info(f"Retrieved document: {document.text[:100]}...")
            logger.info(f"Metadata: {document.metadata}")
        else:
            logger.warning("Document not found")
    except Exception as e:
        logger.warning(f"Could not retrieve document: {e}")
    
    # Bulk operations demonstration
    logger.info("\n📦 Demonstrating bulk operations...")
    
    bulk_texts = [
        "Python is a popular programming language for data science and machine learning.",
        "JavaScript is widely used for web development and frontend applications.",
        "SQL is essential for database management and data querying.",
        "Git is a version control system used by developers worldwide.",
        "Docker enables containerization for consistent application deployment."
    ]
    
    bulk_metadata = [
        {"category": "programming", "language": "python"},
        {"category": "programming", "language": "javascript"},
        {"category": "database", "language": "sql"},
        {"category": "tools", "language": "git"},
        {"category": "devops", "language": "docker"}
    ]
    
    with PerformanceBenchmark("Bulk Upsert") as benchmark:
        bulk_result = await vector_service.bulk_upsert_with_embedding(
            texts=bulk_texts,
            embedding_service=embedding_service,
            metadata_list=bulk_metadata,
            job_id="demo_bulk_upsert"
        )
    
    logger.info(f"✅ Bulk upserted {bulk_result['upserted_count']} documents in {benchmark.elapsed_time:.2f}ms")
    
    # Search the new documents
    logger.info("\n🔍 Searching bulk-inserted documents...")
    
    search_response = await vector_service.search_by_text(
        text="programming languages",
        embedding_service=embedding_service,
        top_k=3,
        include_metadata=True
    )
    
    logger.info(f"Found {len(search_response.results)} programming-related documents:")
    for i, result in enumerate(search_response.results, 1):
        logger.info(f"  {i}. {result.text[:80]}... (score: {result.score:.3f})")
        logger.info(f"     Language: {result.metadata.get('language', 'N/A')}")
    
    # Performance statistics
    logger.info("\n📊 Performance Statistics:")
    
    perf_stats = vector_service.get_performance_stats()
    logger.info(f"Active jobs: {perf_stats['active_jobs']}")
    logger.info(f"Completed jobs: {perf_stats['completed_jobs']}")
    logger.info(f"Failed jobs: {perf_stats['failed_jobs']}")
    logger.info(f"Total items processed: {perf_stats['total_items_processed']}")
    logger.info(f"Supported metrics: {', '.join(perf_stats['supported_metrics'])}")
    
    # Job tracking demonstration
    logger.info("\n📋 Job Tracking:")
    
    active_jobs = vector_service.get_active_jobs()
    for job in active_jobs:
        logger.info(f"Job {job.job_id}: {job.operation} - {job.status}")
        logger.info(f"  Processed: {job.processed_items}/{job.total_items}")
        if job.error_message:
            logger.info(f"  Error: {job.error_message}")
    
    # Cleanup demonstration (optional)
    logger.info("\n🧹 Cleanup demonstration (skipped for safety)")
    logger.info("To clean up, you can use:")
    logger.info("  - vector_service.delete_documents(['doc_1', 'doc_2', ...])")
    logger.info("  - vector_service.delete_index('index_name')")
    
    logger.info("\n✅ Vector Store Service Demonstration Complete!")

async def demonstrate_index_management():
    """Demonstrate index management operations"""
    
    logger.info("🗄️ Starting Index Management Demonstration")
    
    config = get_config()
    vector_provider = WeaviateVectorStoreProvider(config=config)
    vector_service = VectorService(provider=vector_provider)
    
    # List indexes
    logger.info("📋 Current indexes:")
    try:
        indexes = await vector_service.list_indexes()
        for index in indexes:
            logger.info(f"  - {index.name} (dimension: {index.dimension}, metric: {index.metric})")
            
            # Get stats for each index
            try:
                stats = await vector_service.get_index_stats(index.name)
                logger.info(f"    Stats: {stats.total_vector_count} vectors, {stats.index_fullness:.2%} fullness")
            except Exception as e:
                logger.warning(f"    Could not get stats: {e}")
                
    except Exception as e:
        logger.error(f"Could not list indexes: {e}")
    
    # Supported metrics
    logger.info(f"\n📏 Supported distance metrics: {', '.join(vector_provider.get_supported_metrics())}")
    
    logger.info("✅ Index Management Demonstration Complete!")

async def main():
    """Main demonstration function"""
    
    logger.info("🎯 AI Utils - Vector Store Service Demonstration")
    logger.info("=" * 60)
    
    try:
        # Check configuration
        config = get_config()
        logger.info(f"Configuration loaded: Weaviate index '{config.weaviate.index_name}'")
        logger.info(f"Embedding model: {config.openai.embedding_model}")
        logger.info(f"Vector dimension: {config.weaviate.dimension}")
        
        # Check if we should use existing index
        vector_provider = WeaviateVectorStoreProvider(config=config)
        vector_service = VectorService(provider=vector_provider)
        existing_indexes = await vector_service.list_indexes()
        if existing_indexes:
            existing_index = existing_indexes[0]
            logger.info(f"Found existing index: {existing_index.name} (dimension: {existing_index.dimension})")
            # Update config to use existing index
            config.weaviate.index_name = existing_index.name
            config.weaviate.dimension = existing_index.dimension
            logger.info(f"Using existing index: {config.weaviate.index_name}")
        
        # Run demonstrations
        await demonstrate_index_management()
        print()
        await demonstrate_vector_store_operations()
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        logger.error("Please check your configuration and API keys.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 