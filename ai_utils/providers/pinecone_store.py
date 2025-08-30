"""
Pinecone vector store provider implementation.
Implements the VectorStoreProvider interface with Pinecone's vector database.
"""

import asyncio
import logging
import time

from pinecone import Pinecone, ServerlessSpec

from ..config import get_config
from ..interfaces.vector_store import VectorStoreProvider
from ..models import (
    IndexConfig,
    IndexStats,
    ProcessingJob,
    ProcessingStatus,
    VectorDocument,
    VectorQuery,
    VectorSearchResponse,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)


class PineconeVectorStoreProvider(VectorStoreProvider):
    """Pinecone vector store provider implementation"""

    def __init__(self, config=None):
        """Initialize Pinecone vector store provider"""
        self.config = config or get_config()
        self.client = Pinecone(api_key=self.config.pinecone.api_key)
        self._index_cache = {}
        self._supported_metrics = ["cosine", "euclidean", "dotproduct"]

    async def _get_index(self):
        """
        Get or create Pinecone index.

        Returns:
            Pinecone index instance
        """
        index_name = self.config.pinecone.index_name

        if index_name not in self._index_cache:
            try:
                # Check if index exists
                if index_name not in self.client.list_indexes().names():
                    # Create index if it doesn't exist
                    self._create_index(index_name)

                # Connect to index
                self._index_cache[index_name] = self.client.Index(index_name)
                logger.info(f"Connected to Pinecone index: {index_name}")

            except Exception as e:
                logger.error(f"Failed to get index {index_name}: {str(e)}")
                raise

        return self._index_cache[index_name]

    def _create_index(self, index_name: str):
        """
        Create a new Pinecone index.

        Args:
            index_name: Name of the index to create
        """
        try:
            dimension = self.config.pinecone.dimension
            metric = self.config.pinecone.metric

            # Create index with serverless spec for free tier
            # Use cloud and region from config instead of hardcoded values
            cloud = getattr(self.config.pinecone, "cloud", "aws")
            region = getattr(self.config.pinecone, "region", "us-east-1")

            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

            logger.info(
                f"Created Pinecone index: {index_name} (dimension: {dimension}, metric: {metric}, cloud: {cloud}, region: {region})"
            )

        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            raise

    async def upsert_documents(self, documents: list[VectorDocument]) -> ProcessingJob:
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
            status=ProcessingStatus.PROCESSING,
        )

        try:
            index = await self._get_index()

            # Prepare vectors for upsert
            vectors = []
            for doc in documents:
                vector_data = {
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": {
                        **doc.metadata,
                        "text": doc.text,
                        "created_at": doc.created_at.isoformat(),
                    },
                }
                vectors.append(vector_data)

            # Upsert in batches
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                index.upsert(vectors=batch)
                total_upserted += len(batch)
                logger.debug(
                    f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors"
                )

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
            index = await self._get_index()

            # Prepare search parameters
            search_params = {
                "vector": query.embedding,
                "top_k": query.top_k,
                "include_metadata": query.include_metadata,
                "include_values": query.include_embeddings,
            }

            if query.filters:
                search_params["filter"] = query.filters

            # Perform search
            start_time = asyncio.get_event_loop().time()
            search_result = index.query(**search_params)
            search_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Convert results to our model format
            results = []
            for match in search_result.matches:
                result = VectorSearchResult(
                    id=match.id,
                    text=match.metadata.get("text", ""),
                    score=match.score,
                    metadata={k: v for k, v in match.metadata.items() if k != "text"},
                    embedding=match.values if query.include_embeddings else None,
                )
                results.append(result)

            response = VectorSearchResponse(
                results=results,
                query=query.query,
                total_results=len(results),
                search_time_ms=search_time,
            )

            logger.info(
                f"Search completed in {search_time:.2f}ms, found {len(results)} results"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to search similar vectors: {str(e)}")
            raise

    async def delete_documents(self, document_ids: list[str]) -> bool:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            index = await self._get_index()

            # Delete in batches
            batch_size = 100
            total_deleted = 0

            for i in range(0, len(document_ids), batch_size):
                batch_ids = document_ids[i : i + batch_size]
                index.delete(ids=batch_ids)
                total_deleted += len(batch_ids)
                logger.debug(
                    f"Deleted batch {i // batch_size + 1}: {len(batch_ids)} documents"
                )

            logger.info(f"Successfully deleted {total_deleted} documents")

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False

    async def get_document(self, document_id: str) -> VectorDocument | None:
        """
        Get a specific document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        try:
            index = await self._get_index()

            # Fetch document
            fetch_result = index.fetch(ids=[document_id])

            if document_id in fetch_result.vectors:
                vector_data = fetch_result.vectors[document_id]

                # Convert to our model format
                document = VectorDocument(
                    id=vector_data.id,
                    text=vector_data.metadata.get("text", ""),
                    embedding=vector_data.values,
                    metadata={
                        k: v for k, v in vector_data.metadata.items() if k != "text"
                    },
                )

                return document

            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None

    async def create_index(self, config: IndexConfig) -> bool:
        """
        Create a new vector index.

        Args:
            config: Index configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use cloud and region from config instead of hardcoded values
            cloud = getattr(self.config.pinecone, "cloud", "aws")
            region = getattr(self.config.pinecone, "region", "us-east-1")

            self.client.create_index(
                name=config.name,
                dimension=config.dimension,
                metric=config.metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

            logger.info(
                f"Created Pinecone index: {config.name} (cloud: {cloud}, region: {region})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create index {config.name}: {str(e)}")
            return False

    async def list_indices(self) -> list[str]:
        """
        List all available indices.

        Returns:
            List of index names
        """
        try:
            indexes = self.client.list_indexes()
            return [index.name for index in indexes]

        except Exception as e:
            logger.error(f"Failed to list indices: {str(e)}")
            return []

    async def list_indexes(self) -> list[IndexConfig]:
        """
        List all available indexes.

        Returns:
            List of index configurations
        """
        try:
            indexes = self.client.list_indexes()
            configs = []
            for index in indexes:
                replicas = getattr(index, "replicas", 1) or 1
                config = IndexConfig(
                    name=index.name,
                    dimension=index.dimension,
                    metric=index.metric,
                    replicas=replicas,
                )
                configs.append(config)
            return configs
        except Exception as e:
            logger.error(f"Failed to list indexes: {str(e)}")
            return []

    async def get_index_stats(self, index_name: str) -> IndexStats | None:
        """
        Get statistics for a vector index.

        Args:
            index_name: Name of the index

        Returns:
            Index statistics if found, None otherwise
        """
        try:
            # Always use the provided index_name
            index = await self._get_index()  # Only supports default index for now
            desc = index.describe_index_stats()
            stats = IndexStats(
                total_vector_count=desc.total_vector_count,
                dimension=desc.dimension,
                index_fullness=desc.index_fullness,
                last_updated=desc.last_updated,
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return None

    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from cache if present
            if index_name in self._index_cache:
                del self._index_cache[index_name]

            # Delete index
            self.client.delete_index(index_name)

            logger.info(f"Successfully deleted index: {index_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {str(e)}")
            return False

    def get_supported_metrics(self) -> list[str]:
        """
        Get list of supported distance metrics.

        Returns:
            List of supported metrics
        """
        return self._supported_metrics.copy()

    async def health_check(self) -> bool:
        """
        Check if the Pinecone provider is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to list indexes
            self.client.list_indexes()
            return True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {str(e)}")
            return False

    async def clear_cache(self):
        """Clear the index cache"""
        self._index_cache.clear()
        logger.info("Cleared index cache")
