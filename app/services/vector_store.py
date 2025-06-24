from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
from datetime import datetime
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    SearchRequest, SparseVector, NamedVector, NamedSparseVector,
    SparseVectorParams, SparseIndexParams, CreateCollection,
    OptimizersConfigDiff, CollectionInfo, UpdateResult,
    MatchValue, MatchAny, QueryResponse
)

from config import settings
from models import DocumentChunk, CollectionStats


class VectorStore:
    """Qdrant vector store with hybrid search capabilities."""
    
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            prefer_grpc=True,
            grpc_port=settings.qdrant_grpc_port,
            check_compatibility=False  # Suppress version warning
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.embedding_dimension
        
    async def initialize(self):
        """Initialize the vector store and create collection if needed."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                await self._create_collection()
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def _create_collection(self):
        """Create a new collection with hybrid search support."""
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False
                    )
                )
            },
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000
            )
        )
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            info = await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to the vector store."""
        try:
            points = []
            
            for doc in documents:
                # Create point with both dense and sparse vectors
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector={
                        "dense": doc.embeddings,
                        "sparse": SparseVector(
                            indices=doc.sparse_indices,
                            values=doc.sparse_values
                        )
                    },
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "chunk_id": doc.id,
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Batch upload
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    async def search_dense(self, query_embedding: List[float], 
                          top_k: int = 10, 
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)
            
            # Search
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=NamedVector(
                    name="dense",
                    vector=query_embedding
                ),
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )
            
            # Convert to standard format
            search_results = []
            for result in results:
                search_results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_id": result.payload.get("chunk_id", "")
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise
    
    async def search_sparse(self, sparse_indices: List[int], 
                           sparse_values: List[float],
                           top_k: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform sparse vector search."""
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)
            
            # Search with sparse vector
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                ),
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )
            
            # Convert to standard format
            search_results = []
            for result in results:
                search_results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_id": result.payload.get("chunk_id", "")
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            raise
    
    async def search_hybrid(self, query_embedding: List[float],
                           sparse_indices: List[int],
                           sparse_values: List[float],
                           top_k: int = 10,
                           alpha: float = 0.7,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Args:
            alpha: Weight for dense search (0.0 to 1.0)
                   alpha=1.0 means pure dense search
                   alpha=0.0 means pure sparse search
        """
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)
            
            # Perform both searches in parallel
            dense_task = self.search_dense(query_embedding, top_k * 2, filters)
            sparse_task = self.search_sparse(sparse_indices, sparse_values, top_k * 2, filters)
            
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            # Apply Reciprocal Rank Fusion (RRF)
            results = self._apply_rrf_fusion_from_separate_results(
                dense_results, sparse_results, alpha, top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense search
            return await self.search_dense(query_embedding, top_k, filters)
    
    def _apply_rrf_fusion_from_separate_results(self, 
                                                dense_results: List[Dict[str, Any]], 
                                                sparse_results: List[Dict[str, Any]], 
                                                alpha: float, 
                                                top_k: int) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine results from separate dense and sparse searches.
        """
        # Create rank maps for both result sets
        dense_ranks = {result["chunk_id"]: idx + 1 for idx, result in enumerate(dense_results)}
        sparse_ranks = {result["chunk_id"]: idx + 1 for idx, result in enumerate(sparse_results)}
        
        # Combine all unique results
        result_map = {}
        
        # Process dense results
        for result in dense_results:
            chunk_id = result["chunk_id"]
            result_map[chunk_id] = {
                "id": result["id"],
                "content": result["content"],
                "metadata": result["metadata"],
                "chunk_id": chunk_id,
                "dense_rank": dense_ranks.get(chunk_id, len(dense_results) + 1),
                "sparse_rank": sparse_ranks.get(chunk_id, len(sparse_results) + 1),
                "dense_score": result["score"],
                "sparse_score": 0.0
            }
        
        # Process sparse results
        for result in sparse_results:
            chunk_id = result["chunk_id"]
            if chunk_id in result_map:
                result_map[chunk_id]["sparse_score"] = result["score"]
            else:
                result_map[chunk_id] = {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "chunk_id": chunk_id,
                    "dense_rank": dense_ranks.get(chunk_id, len(dense_results) + 1),
                    "sparse_rank": sparse_ranks.get(chunk_id, len(sparse_results) + 1),
                    "dense_score": 0.0,
                    "sparse_score": result["score"]
                }
        
        # Calculate RRF scores
        k = 60  # RRF constant
        results = []
        
        for item in result_map.values():
            # RRF formula with alpha weighting
            rrf_score = (
                alpha * (1.0 / (k + item["dense_rank"])) +
                (1 - alpha) * (1.0 / (k + item["sparse_rank"]))
            )
            
            results.append({
                "id": item["id"],
                "score": rrf_score,
                "content": item["content"],
                "metadata": item["metadata"],
                "chunk_id": item["chunk_id"]
            })
        
        # Sort by RRF score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle list values with MatchAny
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchAny(any=value)
                    )
                )
            else:
                # Handle single values
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    async def get_collection_stats(self) -> CollectionStats:
        """Get collection statistics."""
        try:
            info = await self.client.get_collection(self.collection_name)
            
            # Calculate approximate size
            vectors_count = info.vectors_count or 0
            points_count = info.points_count or 0
            
            # Rough estimate: embedding size * 4 bytes per float * number of vectors
            # Plus metadata overhead
            size_mb = (vectors_count * self.embedding_dimension * 4) / (1024 * 1024)
            size_mb += (points_count * 1024) / (1024 * 1024)  # ~1KB per metadata
            
            return CollectionStats(
                collection_name=self.collection_name,
                total_documents=points_count,
                total_chunks=points_count,
                index_size_mb=round(size_mb, 2),
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    async def delete_by_filename(self, filename: str) -> int:
        """Delete all documents matching a specific filename."""
        try:
            # Build filter for filename
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=filename)
                    )
                ]
            )
            
            # Get points matching the filename
            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = [point.id for point in scroll_result[0]]
            deleted_count = len(points_to_delete)
            
            if points_to_delete:
                # Delete the points
                await self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=points_to_delete
                )
                logger.info(f"Deleted {deleted_count} chunks for file: {filename}")
            else:
                logger.info(f"No existing chunks found for file: {filename}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents by filename: {e}")
            raise
    
    async def delete_collection(self):
        """Delete the collection (use with caution)."""
        try:
            await self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise