# AI Utilities Vector Store Layer - Interface Guide

This guide demonstrates how to interact with the AI utilities vector store layer through various interfaces including REST APIs, Python clients, and integration examples.

## Table of Contents

1. [REST API Interface](#rest-api-interface)
2. [Python Client Interface](#python-client-interface)
3. [Integration Examples](#integration-examples)
4. [Error Handling](#error-handling)
5. [Performance Monitoring](#performance-monitoring)
6. [Security Considerations](#security-considerations)

---

## REST API Interface

### Base URL
```
http://localhost:8000/api/
```

### Authentication
Currently, the API uses IP-based tracking. For production, consider implementing proper authentication.

### Endpoints

#### 1. Document Operations

##### Upsert Documents
```http
POST /api/vector/documents/
Content-Type: application/json

{
  "documents": [
    {
      "id": "doc_1",
      "text": "Machine learning is a subset of artificial intelligence...",
      "metadata": {
        "category": "ai_technology",
        "topic": "machine_learning"
      }
    }
  ],
  "job_id": "optional_job_id"
}
```

**Response:**
```json
{
  "upserted_count": 1,
  "job_id": "upsert_1234567890",
  "status": "completed",
  "processing_time_ms": 245.67
}
```

##### Search Documents
```http
POST /api/vector/search/
Content-Type: application/json

{
  "query": "What is machine learning?",
  "top_k": 5,
  "filters": {
    "category": "ai_technology"
  },
  "include_metadata": true,
  "include_embeddings": false
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_1",
      "text": "Machine learning is a subset of artificial intelligence...",
      "score": 0.95,
      "metadata": {
        "category": "ai_technology",
        "topic": "machine_learning"
      }
    }
  ],
  "query": "What is machine learning?",
  "total_results": 1,
  "search_time_ms": 45.23
}
```

##### Get Document by ID
```http
GET /api/vector/documents/{document_id}/
```

**Response:**
```json
{
  "id": "doc_1",
  "text": "Machine learning is a subset of artificial intelligence...",
  "embedding": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "category": "ai_technology",
    "topic": "machine_learning"
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

##### Delete Documents
```http
DELETE /api/vector/documents/
Content-Type: application/json

{
  "document_ids": ["doc_1", "doc_2"]
}
```

**Response:**
```json
{
  "deleted_count": 2,
  "status": "success"
}
```

#### 2. Bulk Operations

##### Bulk Upsert with Embedding
```http
POST /api/vector/bulk-upsert/
Content-Type: application/json

{
  "texts": [
    "Python is a popular programming language...",
    "SQL is essential for database management...",
    "Git is a version control system..."
  ],
  "metadata_list": [
    {"category": "programming", "language": "python"},
    {"category": "database", "language": "sql"},
    {"category": "tools", "language": "git"}
  ],
  "job_id": "bulk_upsert_123"
}
```

**Response:**
```json
{
  "upserted_count": 3,
  "job_id": "bulk_upsert_123",
  "status": "completed",
  "processing_time_ms": 1250.45
}
```

#### 3. Index Management

##### Get Index Statistics
```http
GET /api/vector/index/stats/
```

**Response:**
```json
{
  "total_vector_count": 1500,
  "dimension": 3072,
  "index_fullness": 0.75,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

##### List Indices
```http
GET /api/vector/index/list/
```

**Response:**
```json
{
  "indices": ["yt-summariser", "test-index"],
  "active_index": "yt-summariser"
}
```

#### 4. Job Tracking

##### Get Job Status
```http
GET /api/vector/jobs/{job_id}/
```

**Response:**
```json
{
  "job_id": "upsert_1234567890",
  "operation": "document_upsert",
  "status": "completed",
  "total_items": 10,
  "processed_items": 10,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:05Z",
  "error_message": null
}
```

##### Get Active Jobs
```http
GET /api/vector/jobs/active/
```

**Response:**
```json
{
  "active_jobs": [
    {
      "job_id": "bulk_upsert_123",
      "operation": "bulk_upsert_with_embedding",
      "status": "processing",
      "total_items": 100,
      "processed_items": 45
    }
  ]
}
```

#### 5. Health Check

##### Service Health
```http
GET /api/vector/health/
```

**Response:**
```json
{
  "status": "healthy",
  "providers": {
    "embedding": "openai",
    "vector_store": "pinecone"
  },
  "index_stats": {
    "total_vectors": 1500,
    "dimension": 3072
  },
  "performance": {
    "avg_search_time_ms": 45.2,
    "avg_upsert_time_ms": 125.8
  }
}
```

---

## Python Client Interface

### Installation

```bash
pip install requests asyncio
```

### Synchronous Client

```python
import requests
import json
from typing import List, Dict, Any

class VectorStoreClient:
    """Synchronous client for vector store operations"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def upsert_documents(self, documents: List[Dict[str, Any]], job_id: str = None) -> Dict[str, Any]:
        """Upsert documents to vector store"""
        url = f"{self.base_url}/vector/documents/"
        payload = {"documents": documents}
        if job_id:
            payload["job_id"] = job_id
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def search_documents(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for similar documents"""
        url = f"{self.base_url}/vector/search/"
        payload = {
            "query": query,
            "top_k": top_k,
            "include_metadata": True
        }
        if filters:
            payload["filters"] = filters
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get document by ID"""
        url = f"{self.base_url}/vector/documents/{document_id}/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents by IDs"""
        url = f"{self.base_url}/vector/documents/"
        payload = {"document_ids": document_ids}
        response = self.session.delete(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def bulk_upsert(self, texts: List[str], metadata_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Bulk upsert texts with automatic embedding"""
        url = f"{self.base_url}/vector/bulk-upsert/"
        payload = {"texts": texts}
        if metadata_list:
            payload["metadata_list"] = metadata_list
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        url = f"{self.base_url}/vector/jobs/{job_id}/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health"""
        url = f"{self.base_url}/vector/health/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# Usage example
client = VectorStoreClient()

# Upsert documents
documents = [
    {
        "id": "doc_1",
        "text": "Machine learning is a subset of artificial intelligence...",
        "metadata": {"category": "ai_technology"}
    }
]
result = client.upsert_documents(documents)
print(f"Upserted {result['upserted_count']} documents")

# Search documents
search_result = client.search_documents("What is machine learning?", top_k=3)
print(f"Found {len(search_result['results'])} results")

# Bulk upsert
texts = ["Python is popular...", "SQL is essential..."]
metadata_list = [{"category": "programming"}, {"category": "database"}]
bulk_result = client.bulk_upsert(texts, metadata_list)
print(f"Bulk upserted {bulk_result['upserted_count']} documents")
```

### Asynchronous Client

```python
import aiohttp
import asyncio
from typing import List, Dict, Any

class AsyncVectorStoreClient:
    """Asynchronous client for vector store operations"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url
    
    async def upsert_documents(self, documents: List[Dict[str, Any]], job_id: str = None) -> Dict[str, Any]:
        """Upsert documents to vector store"""
        url = f"{self.base_url}/vector/documents/"
        payload = {"documents": documents}
        if job_id:
            payload["job_id"] = job_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
    
    async def search_documents(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for similar documents"""
        url = f"{self.base_url}/vector/search/"
        payload = {
            "query": query,
            "top_k": top_k,
            "include_metadata": True
        }
        if filters:
            payload["filters"] = filters
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
    
    async def bulk_upsert(self, texts: List[str], metadata_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Bulk upsert texts with automatic embedding"""
        url = f"{self.base_url}/vector/bulk-upsert/"
        payload = {"texts": texts}
        if metadata_list:
            payload["metadata_list"] = metadata_list
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

# Usage example
async def main():
    client = AsyncVectorStoreClient()
    
    # Concurrent operations
    tasks = [
        client.search_documents("machine learning"),
        client.search_documents("artificial intelligence"),
        client.search_documents("deep learning")
    ]
    
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"Search {i+1}: Found {len(result['results'])} results")

# Run async client
asyncio.run(main())
```

---

## Integration Examples

### 1. Web Application Integration

```python
# Flask application example
from flask import Flask, request, jsonify
from vector_store_client import VectorStoreClient

app = Flask(__name__)
client = VectorStoreClient()

@app.route('/api/search', methods=['POST'])
def search_documents():
    try:
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 5)
        filters = data.get('filters')
        
        result = client.search_documents(query, top_k, filters)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['POST'])
def add_documents():
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        
        result = client.upsert_documents(documents)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. Django Integration

```python
# Django views example
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from vector_store_client import VectorStoreClient

client = VectorStoreClient()

@csrf_exempt
@require_http_methods(["POST"])
def search_view(request):
    try:
        data = json.loads(request.body)
        query = data.get('query')
        top_k = data.get('top_k', 5)
        filters = data.get('filters')
        
        result = client.search_documents(query, top_k, filters)
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upsert_view(request):
    try:
        data = json.loads(request.body)
        documents = data.get('documents', [])
        
        result = client.upsert_documents(documents)
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### 3. Streamlit Application

```python
# Streamlit application example
import streamlit as st
from vector_store_client import VectorStoreClient

client = VectorStoreClient()

st.title("Vector Store Search Interface")

# Search interface
query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results:", 1, 20, 5)

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            try:
                results = client.search_documents(query, top_k)
                
                st.success(f"Found {len(results['results'])} results")
                
                for i, result in enumerate(results['results'], 1):
                    with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                        st.write(f"**Text:** {result['text']}")
                        st.write(f"**Metadata:** {result['metadata']}")
                        
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

# Document upload interface
st.header("Upload Documents")
uploaded_file = st.file_uploader("Upload JSON file with documents", type=['json'])

if uploaded_file:
    try:
        documents = json.load(uploaded_file)
        if st.button("Upload Documents"):
            with st.spinner("Uploading documents..."):
                result = client.upsert_documents(documents)
                st.success(f"Uploaded {result['upserted_count']} documents")
                
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
```

### 4. Jupyter Notebook Integration

```python
# Jupyter notebook example
import pandas as pd
from vector_store_client import VectorStoreClient

client = VectorStoreClient()

# Load data from CSV
df = pd.read_csv('documents.csv')

# Bulk upsert documents
texts = df['text'].tolist()
metadata_list = df[['category', 'topic']].to_dict('records')

result = client.bulk_upsert(texts, metadata_list)
print(f"Uploaded {result['upserted_count']} documents")

# Search and analyze results
queries = ["machine learning", "artificial intelligence", "deep learning"]
results = []

for query in queries:
    search_result = client.search_documents(query, top_k=10)
    results.append({
        'query': query,
        'results_count': len(search_result['results']),
        'avg_score': sum(r['score'] for r in search_result['results']) / len(search_result['results'])
    })

results_df = pd.DataFrame(results)
results_df
```

---

## Error Handling

### Common Error Responses

```json
{
  "error": "Invalid request format",
  "status_code": 400,
  "details": "Missing required field 'query'"
}
```

```json
{
  "error": "Vector store connection failed",
  "status_code": 503,
  "details": "Pinecone API timeout"
}
```

```json
{
  "error": "Embedding service error",
  "status_code": 500,
  "details": "OpenAI API rate limit exceeded"
}
```

### Error Handling in Python Client

```python
import requests
from requests.exceptions import RequestException

class VectorStoreClient:
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _handle_response(self, response):
        """Handle API response and errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                raise ValueError(f"Bad request: {response.json().get('details', str(e))}")
            elif response.status_code == 404:
                raise FileNotFoundError(f"Resource not found: {response.json().get('details', str(e))}")
            elif response.status_code == 503:
                raise ConnectionError(f"Service unavailable: {response.json().get('details', str(e))}")
            else:
                raise Exception(f"API error: {response.json().get('details', str(e))}")
        except RequestException as e:
            raise ConnectionError(f"Network error: {str(e)}")
    
    def search_documents(self, query: str, top_k: int = 5):
        """Search with proper error handling"""
        try:
            url = f"{self.base_url}/vector/search/"
            payload = {"query": query, "top_k": top_k}
            response = self.session.post(url, json=payload)
            return self._handle_response(response)
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return {"results": [], "error": str(e)}
```

---

## Performance Monitoring

### Response Time Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor API performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1000
            print(f"{func.__name__} completed in {elapsed_time:.2f}ms")
            return result
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            print(f"{func.__name__} failed after {elapsed_time:.2f}ms: {str(e)}")
            raise
    return wrapper

# Apply to client methods
class MonitoredVectorStoreClient(VectorStoreClient):
    @monitor_performance
    def search_documents(self, query: str, top_k: int = 5):
        return super().search_documents(query, top_k)
    
    @monitor_performance
    def upsert_documents(self, documents, job_id=None):
        return super().upsert_documents(documents, job_id)
```

### Batch Processing

```python
def batch_process_documents(documents, batch_size=100):
    """Process documents in batches for better performance"""
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            result = client.upsert_documents(batch)
            results.append(result)
            print(f"Processed batch {i//batch_size + 1}: {len(batch)} documents")
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {str(e)}")
    
    return results
```

---

## Security Considerations

### API Key Management

```python
import os
from dotenv import load_dotenv

load_dotenv()

class SecureVectorStoreClient(VectorStoreClient):
    def __init__(self, base_url: str = None):
        # Use environment variables for configuration
        api_key = os.getenv('VECTOR_STORE_API_KEY')
        if not api_key:
            raise ValueError("VECTOR_STORE_API_KEY environment variable is required")
        
        # Add authentication headers
        super().__init__(base_url or os.getenv('VECTOR_STORE_URL'))
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
```

### Rate Limiting

```python
import time
from collections import deque

class RateLimitedVectorStoreClient(VectorStoreClient):
    def __init__(self, requests_per_minute=60):
        super().__init__()
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()
        
        # Remove old requests
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Add current request
        self.request_times.append(now)
    
    def search_documents(self, query: str, top_k: int = 5):
        self._check_rate_limit()
        return super().search_documents(query, top_k)
```

---

## Configuration Examples

### Environment Variables

```bash
# .env file
VECTOR_STORE_URL=http://localhost:8000/api
VECTOR_STORE_API_KEY=your-api-key
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=yt-summariser
```

### Configuration Files

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class VectorStoreConfig:
    base_url: str = os.getenv('VECTOR_STORE_URL', 'http://localhost:8000/api')
    api_key: str = os.getenv('VECTOR_STORE_API_KEY')
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100

# Usage
config = VectorStoreConfig()
client = VectorStoreClient(config.base_url)
```

---

## Testing Examples

### Unit Tests

```python
import unittest
from unittest.mock import Mock, patch
from vector_store_client import VectorStoreClient

class TestVectorStoreClient(unittest.TestCase):
    def setUp(self):
        self.client = VectorStoreClient()
    
    @patch('requests.Session.post')
    def test_search_documents_success(self, mock_post):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{"id": "doc1", "text": "test", "score": 0.9}],
            "total_results": 1
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.search_documents("test query")
        
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["score"], 0.9)
    
    @patch('requests.Session.post')
    def test_search_documents_error(self, mock_post):
        # Mock error response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        with self.assertRaises(Exception):
            self.client.search_documents("test query")

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
import pytest
from vector_store_client import VectorStoreClient

@pytest.fixture
def client():
    return VectorStoreClient()

def test_end_to_end_workflow(client):
    """Test complete workflow: upsert -> search -> delete"""
    
    # 1. Upsert documents
    documents = [
        {
            "id": "test_doc_1",
            "text": "This is a test document about machine learning",
            "metadata": {"category": "test"}
        }
    ]
    
    upsert_result = client.upsert_documents(documents)
    assert upsert_result["upserted_count"] == 1
    
    # 2. Search documents
    search_result = client.search_documents("machine learning")
    assert len(search_result["results"]) > 0
    assert search_result["results"][0]["id"] == "test_doc_1"
    
    # 3. Delete documents
    delete_result = client.delete_documents(["test_doc_1"])
    assert delete_result["deleted_count"] == 1
```

This interface guide provides comprehensive examples of how to interact with the AI utilities vector store layer through various interfaces. The examples cover REST APIs, Python clients, integration patterns, error handling, performance monitoring, and security considerations. 