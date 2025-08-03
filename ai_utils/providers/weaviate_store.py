"""
Weaviate vector store provider implementation.
Implements the VectorStoreProvider interface with Weaviate's vector database.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import weaviate
import weaviate.classes as wvc
from ..interfaces.vector_store import VectorStoreProvider
from ..models import (
    VectorDocument, VectorQuery, VectorSearchResult, VectorSearchResponse,
    IndexConfig, IndexStats, ProcessingJob, ProcessingStatus
)
from ..config import get_config
from ..utils.performance import PerformanceBenchmark

logger = logging.getLogger(__name__)

class WeaviateVectorStoreProvider(VectorStoreProvider):
    """Weaviate vector store provider implementation"""
    
    def __init__(self, config=None):
        """Initialize Weaviate vector store provider"""
        self.config = config or get_config()
        self.client = None
        self.collection = None
        self._supported_metrics = ["cosine", "dot", "l2-squared", "hamming", "manhattan"]
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Weaviate client and connection"""
        try:
            weaviate_config = self.config.weaviate
            
            # Determine connection URL
            url = weaviate_config.cluster_url or weaviate_config.url
            
            # Set up authentication
            auth_config = None
            if weaviate_config.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_config.api_key)
            elif weaviate_config.auth_client_secret:
                auth_config = weaviate.auth.AuthClientCredentials(
                    client_secret=weaviate_config.auth_client_secret
                )
            
            # Create client using the new v4 API
            if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
                # For local connections
                self.client = weaviate.connect_to_local(
                    host=url.replace("http://", "").replace("https://", "").split(":")[0],
                    port=int(url.split(":")[-1]) if ":" in url.split("//")[1] else 8080,
                    headers=weaviate_config.headers or {}
                )
            elif ".weaviate.cloud" in url:
                # For Weaviate Cloud connections
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=auth_config,
                    headers=weaviate_config.headers or {},
                    additional_config=weaviate.config.AdditionalConfig(
                        timeout=weaviate.config.Timeout(
                            init=weaviate_config.timeout,
                            query=weaviate_config.timeout,
                            insert=weaviate_config.timeout * 2
                        )
                    )
                )
            else:
                # For custom/remote connections
                host = url.replace("http://", "").replace("https://", "")
                port = 443 if url.startswith("https://") else 8080
                if ":" in host:
                    host, port_str = host.split(":")
                    port = int(port_str)
                    
                self.client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=url.startswith("https://"),
                    grpc_host=host,
                    grpc_port=port + 1,  # gRPC typically uses port + 1
                    grpc_secure=url.startswith("https://"),
                    auth_credentials=auth_config,
                    headers=weaviate_config.headers or {},
                    additional_config=weaviate.config.AdditionalConfig(
                        timeout=weaviate.config.Timeout(
                            init=weaviate_config.timeout,
                            query=weaviate_config.timeout,
                            insert=weaviate_config.timeout * 2
                        )
                    )
                )
            
            # Test connection
            if self.client.is_ready():
                logger.info(f"Connected to Weaviate at {url}")
                self._ensure_collection_exists()
            else:
                raise ConnectionError("Failed to connect to Weaviate")
                
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {str(e)}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        try:
            collection_name = self.config.weaviate.collection_name
            
            # Check if collection exists
            if not self.client.collections.exists(collection_name):
                logger.info(f"Creating Weaviate collection: {collection_name}")
                
                try:
                    # Define collection schema using v4 API
                    self.client.collections.create(
                        collection_name,
                        properties=[
                            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                            wvc.config.Property(name="created_at", data_type=wvc.config.DataType.DATE),
                            wvc.config.Property(name="original_id", data_type=wvc.config.DataType.TEXT),
                            # Define metadata as individual properties to satisfy Weaviate v4 requirements
                            wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                            wvc.config.Property(name="video_id", data_type=wvc.config.DataType.TEXT),
                            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                            wvc.config.Property(name="channel", data_type=wvc.config.DataType.TEXT),
                            wvc.config.Property(name="duration", data_type=wvc.config.DataType.NUMBER),
                            wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.NUMBER),
                            wvc.config.Property(name="start_time", data_type=wvc.config.DataType.NUMBER),
                            wvc.config.Property(name="end_time", data_type=wvc.config.DataType.NUMBER),
                            wvc.config.Property(name="sequence_number", data_type=wvc.config.DataType.NUMBER),
                            wvc.config.Property(name="youtube_url", data_type=wvc.config.DataType.TEXT),
                            wvc.config.Property(name="key_points_count", data_type=wvc.config.DataType.NUMBER)
                        ],
                        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_weaviate(),
                        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                            distance_metric=self._get_weaviate_distance_metric()
                        )
                    )
                    logger.info(f"Successfully created collection: {collection_name}")
                except Exception as create_error:
                    # If collection creation fails, log the error but continue
                    # This is common in Weaviate Cloud where collections might be pre-created
                    logger.warning(f"Failed to create collection, but continuing: {create_error}")
            else:
                logger.info(f"Collection '{collection_name}' already exists")
            
            # Get collection reference
            self.collection = self.client.collections.get(collection_name)
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            # Make sure collection is set to None on error
            self.collection = None
            raise
    
    def _get_weaviate_distance_metric(self):
        """Convert config distance metric to Weaviate format"""
        metric_mapping = {
            "cosine": wvc.config.VectorDistances.COSINE,
            "dot": wvc.config.VectorDistances.DOT,
            "euclidean": wvc.config.VectorDistances.L2_SQUARED,
            "l2": wvc.config.VectorDistances.L2_SQUARED,
            "manhattan": wvc.config.VectorDistances.MANHATTAN,
            "hamming": wvc.config.VectorDistances.HAMMING
        }
        config_metric = self.config.weaviate.distance_metric.lower()
        return metric_mapping.get(config_metric, wvc.config.VectorDistances.COSINE)
    
    def _get_weaviate_distance_metric_for_config(self, metric: str):
        """Convert any distance metric to Weaviate format"""
        metric_mapping = {
            "cosine": wvc.config.VectorDistances.COSINE,
            "dot": wvc.config.VectorDistances.DOT,
            "euclidean": wvc.config.VectorDistances.L2_SQUARED,
            "l2": wvc.config.VectorDistances.L2_SQUARED,
            "l2-squared": wvc.config.VectorDistances.L2_SQUARED,
            "manhattan": wvc.config.VectorDistances.MANHATTAN,
            "hamming": wvc.config.VectorDistances.HAMMING
        }
        metric_lower = metric.lower() if metric else "cosine"
        return metric_mapping.get(metric_lower, wvc.config.VectorDistances.COSINE)
    
    async def upsert_documents(self, documents: List[VectorDocument]) -> ProcessingJob:
        """
        Upsert documents into the vector store.
        
        Args:
            documents: List of documents to upsert
            
        Returns:
            Processing job for tracking the operation
        """
        job = ProcessingJob(
            job_id=f"upsert_{int(time.time())}",
            operation="document_upsert",
            total_items=len(documents),
            status=ProcessingStatus.PROCESSING
        )
        
        try:
            if self.collection is None:
                self._ensure_collection_exists()
                if self.collection is None:
                    raise RuntimeError("Collection not initialized")
            
            # Prepare objects for batch insertion using v4 API
            objects = []
            for doc in documents:
                # Flatten metadata fields to match the new schema
                obj_data = {
                    "text": doc.text,
                    "created_at": doc.created_at.isoformat() + "Z" if not doc.created_at.tzinfo else doc.created_at.isoformat(),
                    # Extract metadata fields as individual properties
                    "type": doc.metadata.get("type", ""),
                    "video_id": doc.metadata.get("video_id", ""),
                    "title": doc.metadata.get("title", ""),
                    "channel": doc.metadata.get("channel", ""),
                    "duration": doc.metadata.get("duration", 0),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "start_time": doc.metadata.get("start_time", 0),
                    "end_time": doc.metadata.get("end_time", 0),
                    "sequence_number": doc.metadata.get("sequence_number", 0),
                    "youtube_url": doc.metadata.get("youtube_url", ""),
                    "key_points_count": doc.metadata.get("key_points_count", 0)
                }
                # Use UUID format for Weaviate v4, store original ID in properties
                import uuid as uuid_lib
                weaviate_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, doc.id))
                obj_data["original_id"] = doc.id  # Store original ID as property
                
                objects.append(wvc.data.DataObject(
                    properties=obj_data,
                    uuid=weaviate_uuid
                    # No vector parameter - Weaviate will generate embeddings natively
                ))
            
            # Insert in batches using v4 batch API
            batch_size = self.config.weaviate.batch_size
            total_upserted = 0
            
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                
                # Use the new batch insert API
                response = self.collection.data.insert_many(batch)
                
                # Count successful inserts
                successful_inserts = len([obj for obj in response.uuids.values() if obj is not None])
                total_upserted += successful_inserts
            
            logger.info(f"Successfully upserted {total_upserted} documents")
            
            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = total_upserted
            job.updated_at = time.time()
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = time.time()
            raise
    
    async def search_similar(self, query: VectorQuery) -> VectorSearchResponse:
        """
        Search for similar documents.
        
        Args:
            query: Search query with text and parameters
            
        Returns:
            Search response with results and metadata
        """
        try:
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            
            start_time = time.time()
            
            # Use text-based search with native vectorizer instead of near_vector
            # Correct syntax for Weaviate v4 near_text search - request distance and certainty
            search_kwargs = {
                "limit": query.top_k,
                "return_metadata": wvc.query.MetadataQuery(distance=True, certainty=True)
            }
            
            # Add filters if provided
            where_filter = None
            if query.filters:
                where_filter = self._build_where_filter(query.filters)
            
            # Execute search using near_text for native vectorization
            if where_filter:
                response = self.collection.query.near_text(
                    query=query.query,
                    filters=where_filter,
                    **search_kwargs
                )
            else:
                response = self.collection.query.near_text(
                    query=query.query,
                    **search_kwargs
                )
            search_time = (time.time() - start_time) * 1000
            
            # Convert results to our model format
            results = []
            for obj in response.objects:
                
                # Extract confidence score from Weaviate metadata
                extracted_score = 0.0
                
                if hasattr(obj, 'metadata') and obj.metadata:
                    if hasattr(obj.metadata, 'certainty') and obj.metadata.certainty is not None:
                        # Certainty: 0.0 - 1.0 (higher is better)
                        extracted_score = obj.metadata.certainty
                    elif hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                        # Distance: 0.0 - 2.0 (lower is better), convert to similarity (higher is better)
                        extracted_score = max(0.0, 1.0 - obj.metadata.distance)
                    elif hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                        # Score: if available, use directly
                        extracted_score = obj.metadata.score
                
                # Reconstruct metadata from individual properties
                metadata = {
                    "type": obj.properties.get("type", ""),
                    "video_id": obj.properties.get("video_id", ""),
                    "title": obj.properties.get("title", ""),
                    "channel": obj.properties.get("channel", ""),
                    "duration": obj.properties.get("duration", 0),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "start_time": obj.properties.get("start_time", 0),
                    "end_time": obj.properties.get("end_time", 0),
                    "sequence_number": obj.properties.get("sequence_number", 0),
                    "youtube_url": obj.properties.get("youtube_url", ""),
                    "key_points_count": obj.properties.get("key_points_count", 0)
                }
                
                result = VectorSearchResult(
                    id=obj.properties.get("original_id", str(obj.uuid)),  # Use original ID if available
                    text=obj.properties.get("text", ""),
                    score=extracted_score,
                    metadata=metadata,
                    embedding=None  # Vector retrieval would need separate API call in v4
                )
                results.append(result)
            
            search_response = VectorSearchResponse(
                results=results,
                query=query.query,
                total_results=len(results),
                search_time_ms=search_time
            )
            
            logger.info(f"Search completed in {search_time:.2f}ms, found {len(results)} results")
            
            return search_response
            
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {str(e)}")
            raise
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[wvc.query.Filter]:
        """Build Weaviate where filter from dictionary"""
        try:
            # Simple implementation using v4 API - can be extended for more complex filters
            conditions = []
            
            for key, value in filters.items():
                if isinstance(value, str):
                    # Direct property access since metadata is now flattened
                    conditions.append(wvc.query.Filter.by_property(key).equal(value))
                elif isinstance(value, (int, float)):
                    conditions.append(wvc.query.Filter.by_property(key).equal(value))
                elif isinstance(value, dict):
                    # Handle range queries, etc.
                    if "gte" in value:
                        conditions.append(wvc.query.Filter.by_property(key).greater_or_equal(value["gte"]))
                    if "lte" in value:
                        conditions.append(wvc.query.Filter.by_property(key).less_or_equal(value["lte"]))
                    if "eq" in value:
                        conditions.append(wvc.query.Filter.by_property(key).equal(value["eq"]))
            
            # Combine conditions with AND
            if len(conditions) == 1:
                return conditions[0]
            elif len(conditions) > 1:
                result = conditions[0]
                for condition in conditions[1:]:
                    result = result & condition
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to build where filter: {str(e)}")
            return None
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            
            # Delete documents by UUID using v4 API
            total_deleted = 0
            batch_size = self.config.weaviate.batch_size
            
            for i in range(0, len(document_ids), batch_size):
                batch_ids = document_ids[i:i + batch_size]
                
                # Use batch delete for better performance
                try:
                    response = self.collection.data.delete_many(
                        where=wvc.query.Filter.by_id().contains_any(batch_ids)
                    )
                    # Count successful deletions (in v4, this might be handled differently)
                    successful_deletes = len(batch_ids)  # Assume all successful unless error
                    total_deleted += successful_deletes
                except Exception as e:
                    logger.warning(f"Failed to delete batch: {str(e)}")
                    # Fall back to individual deletes
                    for doc_id in batch_ids:
                        try:
                            self.collection.data.delete_by_id(doc_id)
                            total_deleted += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete document {doc_id}: {str(e)}")
            
            logger.info(f"Successfully deleted {total_deleted} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        try:
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            
            # Fetch document by UUID using v4 API
            obj = self.collection.query.fetch_object_by_id(
                uuid=document_id
            )
            
            if obj:
                # Reconstruct metadata from individual properties
                metadata = {
                    "type": obj.properties.get("type", ""),
                    "video_id": obj.properties.get("video_id", ""),
                    "title": obj.properties.get("title", ""),
                    "channel": obj.properties.get("channel", ""),
                    "duration": obj.properties.get("duration", 0),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "start_time": obj.properties.get("start_time", 0),
                    "end_time": obj.properties.get("end_time", 0),
                    "sequence_number": obj.properties.get("sequence_number", 0),
                    "youtube_url": obj.properties.get("youtube_url", ""),
                    "key_points_count": obj.properties.get("key_points_count", 0)
                }
                
                # Convert to our model format
                document = VectorDocument(
                    id=obj.properties.get("original_id", str(obj.uuid)),  # Use original ID if available
                    text=obj.properties.get("text", ""),
                    embedding=[],  # Vector retrieval would need separate API call in v4
                    metadata=metadata,
                    created_at=datetime.fromisoformat(obj.properties.get("created_at", datetime.now().isoformat()).replace("Z", "+00:00"))
                )
                return document
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def create_index(self, config: IndexConfig) -> bool:
        """
        Create a new vector index (collection in Weaviate).
        
        Args:
            config: Index configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collections = self.client.collections
            
            # Check if collection already exists
            if collections.exists(config.name):
                logger.warning(f"Collection {config.name} already exists")
                return True
            
            # Create new collection
            collections.create(
                name=config.name,
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="created_at", data_type=wvc.config.DataType.DATE),
                    wvc.config.Property(name="original_id", data_type=wvc.config.DataType.TEXT),
                    # Define metadata as individual properties to satisfy Weaviate v4 requirements
                    wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="video_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="channel", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="duration", data_type=wvc.config.DataType.NUMBER),
                    wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.NUMBER),
                    wvc.config.Property(name="start_time", data_type=wvc.config.DataType.NUMBER),
                    wvc.config.Property(name="end_time", data_type=wvc.config.DataType.NUMBER),
                    wvc.config.Property(name="sequence_number", data_type=wvc.config.DataType.NUMBER),
                    wvc.config.Property(name="youtube_url", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="key_points_count", data_type=wvc.config.DataType.NUMBER)
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_weaviate(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=self._get_weaviate_distance_metric()
                )
            )
            
            logger.info(f"Created Weaviate collection: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {config.name}: {str(e)}")
            return False
    
    async def list_indices(self) -> List[str]:
        """
        List all available indices (collections).
        
        Returns:
            List of collection names
        """
        try:
            schema = self.client.schema.get()
            collection_names = [class_def["class"] for class_def in schema.get("classes", [])]
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    async def list_indexes(self) -> List[IndexConfig]:
        """
        List all available indexes (collections) with configurations.
        
        Returns:
            List of index configurations
        """
        try:
            schema = self.client.schema.get()
            configs = []
            
            for class_def in schema.get("classes", []):
                # Extract configuration from schema
                vector_config = class_def.get("vectorIndexConfig", {})
                
                config = IndexConfig(
                    name=class_def["class"],
                    dimension=vector_config.get("vectorCacheMaxObjects", self.config.weaviate.dimension),
                    metric=vector_config.get("distance", "cosine"),
                    replicas=1  # Weaviate handles replication differently
                )
                configs.append(config)
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to list collections with config: {str(e)}")
            return []
    
    async def get_index_stats(self, index_name: str) -> Optional[IndexStats]:
        """
        Get statistics for a vector index (collection).
        
        Args:
            index_name: Name of the collection
            
        Returns:
            Index statistics if found, None otherwise
        """
        try:
            collection_name = index_name or self.config.weaviate.collection_name
            
            # Get collection stats
            collections = self.client.collections
            if not collections.exists(collection_name):
                return None
            
            collection = collections.get(collection_name)
            
            # Get object count
            result = collection.aggregate.over_all(total_count=True)
            total_count = result.total_count if result.total_count else 0
            
            stats = IndexStats(
                total_vector_count=total_count,
                dimension=self.config.weaviate.dimension,
                index_fullness=0.0,  # Weaviate doesn't provide this metric
                last_updated=datetime.now()
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return None
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index (collection).
        
        Args:
            index_name: Name of the collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collections = self.client.collections
            
            if collections.exists(index_name):
                collections.delete(index_name)
                logger.info(f"Successfully deleted collection: {index_name}")
                return True
            else:
                logger.warning(f"Collection {index_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete collection {index_name}: {str(e)}")
            return False
    
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of supported distance metrics.
        
        Returns:
            List of supported metrics
        """
        return self._supported_metrics.copy()
    
    async def health_check(self) -> bool:
        """
        Check if the Weaviate provider is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            return self.client.is_ready() if self.client else False
        except Exception as e:
            logger.error(f"Weaviate health check failed: {str(e)}")
            return False
    
    async def clear_cache(self):
        """Clear any local caches"""
        # Weaviate client doesn't maintain local caches that need clearing
        logger.info("Weaviate cache cleared (no-op)")