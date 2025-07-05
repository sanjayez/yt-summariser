#!/usr/bin/env python3
"""
Demonstration of the AI Utils embedding service.
Shows how to use the OpenAI embeddings provider and service.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_utils.providers import OpenAIEmbeddingProvider
from ai_utils.services import EmbeddingService
from ai_utils.utils.performance import benchmark_embedding_service, PerformanceBenchmark
from ai_utils.config import get_config

async def demo_basic_embedding():
    """Demonstrate basic embedding functionality"""
    print("🔤 Basic Embedding Demo")
    print("=" * 50)
    
    # Initialize provider and service
    provider = OpenAIEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    
    # Test single text embedding
    text = "Hello, this is a test of the embedding service!"
    print(f"Embedding text: '{text}'")
    
    try:
        response = await service.embed_text(text)
        print(f"✅ Success! Embedding dimension: {len(response.embedding)}")
        print(f"   Model: {response.model}")
        print(f"   Tokens used: {response.usage['prompt_tokens']}")
        print(f"   Request ID: {response.request_id}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

async def demo_batch_embedding():
    """Demonstrate batch embedding functionality"""
    print("\n📦 Batch Embedding Demo")
    print("=" * 50)
    
    # Initialize provider and service
    provider = OpenAIEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    
    # Test batch embedding
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "Natural language processing enables computers to understand human language.",
        "Vector embeddings represent text as numerical vectors."
    ]
    
    print(f"Embedding {len(texts)} texts in batch...")
    
    try:
        response = await service.embed_batch(texts)
        print(f"✅ Success! Generated {len(response.embeddings)} embeddings")
        print(f"   Model: {response.model}")
        print(f"   Total tokens used: {response.usage['prompt_tokens']}")
        print(f"   Average tokens per text: {response.usage['prompt_tokens'] / len(texts):.1f}")
        
        # Show embedding dimensions
        for i, embedding in enumerate(response.embeddings):
            print(f"   Text {i+1}: {len(embedding)} dimensions")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

async def demo_document_embedding():
    """Demonstrate document embedding with metadata"""
    print("\n📄 Document Embedding Demo")
    print("=" * 50)
    
    # Initialize provider and service
    provider = OpenAIEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    
    # Test document embedding
    documents = [
        {
            "id": "doc1",
            "text": "Introduction to machine learning concepts and algorithms.",
            "category": "education",
            "author": "Dr. Smith"
        },
        {
            "id": "doc2", 
            "text": "Advanced techniques in natural language processing.",
            "category": "research",
            "author": "Prof. Johnson"
        },
        {
            "id": "doc3",
            "text": "Practical applications of deep learning in industry.",
            "category": "business",
            "author": "Jane Doe"
        }
    ]
    
    print(f"Embedding {len(documents)} documents with metadata...")
    
    try:
        result = await service.embed_documents(documents)
        print(f"✅ Success! Processed {len(result)} documents")
        
        for doc in result:
            print(f"   Document {doc['id']}: {len(doc['embedding'])} dimensions")
            print(f"     Category: {doc['category']}")
            print(f"     Author: {doc['author']}")
            print(f"     Model: {doc['embedding_model']}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

async def demo_performance_benchmarking():
    """Demonstrate performance benchmarking"""
    print("\n⚡ Performance Benchmarking Demo")
    print("=" * 50)
    
    # Initialize provider and service
    provider = OpenAIEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    
    # Generate test data
    test_texts = [
        "This is a sample text for testing embedding performance.",
        "Machine learning algorithms process large datasets efficiently.",
        "Natural language processing enables text understanding.",
        "Vector embeddings capture semantic meaning in text.",
        "Deep learning models require significant computational resources.",
        "Artificial intelligence transforms various industries.",
        "Data science combines statistics and programming skills.",
        "Neural networks mimic biological brain structures.",
        "Computer vision processes visual information effectively.",
        "Robotics integrates hardware and software systems."
    ] * 5  # Repeat to get more test data
    
    print(f"Running performance benchmarks with {len(test_texts)} texts...")
    
    try:
        benchmark = await benchmark_embedding_service(
            service=service,
            test_texts=test_texts,
            batch_sizes=[1, 5, 10]
        )
        
        # Generate and display report
        report = benchmark.generate_report()
        print("\n📊 Performance Report:")
        print("-" * 30)
        print(report)
        
        # Show summary stats
        summary = benchmark.get_summary_stats()
        print(f"\n📈 Summary:")
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Total items processed: {summary['total_items_processed']}")
        print(f"   Total time: {summary['total_time']:.2f}s")
        print(f"   Average success rate: {summary['average_success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

async def demo_health_check():
    """Demonstrate health checking"""
    print("\n🏥 Health Check Demo")
    print("=" * 50)
    
    # Initialize provider and service
    provider = OpenAIEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    
    print("Checking provider health...")
    
    try:
        # Check provider health
        provider_healthy = await provider.health_check()
        print(f"Provider health: {'✅ Healthy' if provider_healthy else '❌ Unhealthy'}")
        
        # Check service health
        service_healthy = await service.health_check()
        print(f"Service health: {'✅ Healthy' if service_healthy else '❌ Unhealthy'}")
        
        # Get performance stats
        stats = service.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Active jobs: {stats['active_jobs']}")
        print(f"   Completed jobs: {stats['completed_jobs']}")
        print(f"   Failed jobs: {stats['failed_jobs']}")
        print(f"   Total items processed: {stats['total_items_processed']}")
        print(f"   Supported models: {', '.join(stats['supported_models'])}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

async def main():
    """Run all demonstrations"""
    print("🚀 AI Utils Embedding Service Demo")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    config = get_config()
    if not config.openai.api_key:
        print("❌ Error: OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    demos = [
        ("Basic Embedding", demo_basic_embedding),
        ("Batch Embedding", demo_batch_embedding),
        ("Document Embedding", demo_document_embedding),
        ("Performance Benchmarking", demo_performance_benchmarking),
        ("Health Check", demo_health_check)
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\n🎯 Running {name}...")
        try:
            success = await demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Demo Summary:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} demos passed")

if __name__ == "__main__":
    asyncio.run(main()) 