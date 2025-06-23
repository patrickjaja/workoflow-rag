from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict

from services.vector_store import VectorStore
from services.embeddings import EmbeddingService, LLMService
from models import SearchResult
from config import settings


class HybridSearchEngine:
    """Hybrid search engine combining dense and sparse search with reranking."""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = LLMService()
        
    async def search(self, 
                    query: str, 
                    top_k: int = 10,
                    search_type: str = "hybrid",
                    filters: Optional[Dict[str, Any]] = None,
                    rerank: bool = True) -> List[SearchResult]:
        """
        Perform search with specified strategy.
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: "hybrid", "dense", or "sparse"
            filters: Optional metadata filters
            rerank: Whether to apply LLM reranking
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Generate sparse vector for query
            sparse_indices, sparse_values = self._generate_query_sparse_vector(query)
            
            # Perform search based on type
            if search_type == "dense":
                results = await self.vector_store.search_dense(
                    query_embedding, top_k * 2 if rerank else top_k, filters
                )
            elif search_type == "sparse":
                results = await self.vector_store.search_sparse(
                    sparse_indices, sparse_values, top_k * 2 if rerank else top_k, filters
                )
            else:  # hybrid
                results = await self.vector_store.search_hybrid(
                    query_embedding, sparse_indices, sparse_values,
                    top_k * 2 if rerank else top_k,
                    alpha=settings.hybrid_alpha,
                    filters=filters
                )
            
            # Apply reranking if enabled
            if rerank and len(results) > 0:
                results = await self._rerank_results(query, results, top_k)
            else:
                results = results[:top_k]
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                # Extract highlights
                highlights = self._extract_highlights(query, result['content'])
                
                search_results.append(SearchResult(
                    id=result['id'],
                    content=result['content'],
                    score=result['score'],
                    metadata=result['metadata'],
                    highlights=highlights
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _generate_query_sparse_vector(self, query: str) -> tuple[List[int], List[float]]:
        """Generate sparse vector for query."""
        # Simple tokenization
        words = query.lower().split()
        word_counts = defaultdict(int)
        
        for word in words:
            # Simple hash to get index
            index = abs(hash(word)) % 10000
            word_counts[index] += 1
        
        # Convert to sparse format
        indices = list(word_counts.keys())
        values = [float(count) for count in word_counts.values()]
        
        # Normalize
        if values:
            max_val = max(values)
            values = [v / max_val for v in values]
        
        return indices, values
    
    async def _rerank_results(self, query: str, results: List[Dict[str, Any]], 
                             top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder approach with LLM.
        """
        try:
            # For efficiency, only rerank top candidates
            candidates = results[:min(settings.rerank_top_k, len(results))]
            
            # Calculate relevance scores using cosine similarity as base
            for candidate in candidates:
                # You could implement a more sophisticated reranking here
                # For now, we'll use a combination of the original score
                # and keyword matching
                keyword_score = self._calculate_keyword_score(query, candidate['content'])
                candidate['rerank_score'] = (
                    0.7 * candidate['score'] + 
                    0.3 * keyword_score
                )
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # If we have LLM reranking enabled and few enough candidates
            if len(candidates) <= 5:
                candidates = await self.llm_service.rerank_results(query, candidates)
            
            # Take top k
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results if reranking fails
            return results[:top_k]
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword-based relevance score."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _extract_highlights(self, query: str, content: str, 
                           context_words: int = 10) -> List[str]:
        """Extract highlighted snippets from content."""
        highlights = []
        query_words = set(query.lower().split())
        content_words = content.split()
        
        # Find positions of query words in content
        for i, word in enumerate(content_words):
            if word.lower() in query_words:
                # Extract context around the word
                start = max(0, i - context_words)
                end = min(len(content_words), i + context_words + 1)
                
                snippet = ' '.join(content_words[start:end])
                if start > 0:
                    snippet = '...' + snippet
                if end < len(content_words):
                    snippet = snippet + '...'
                
                highlights.append(snippet)
                
                # Limit to 3 highlights
                if len(highlights) >= 3:
                    break
        
        return highlights
    
    async def find_similar(self, document_id: str, top_k: int = 10) -> List[SearchResult]:
        """Find documents similar to a given document."""
        try:
            # First, retrieve the document
            doc_results = await self.vector_store.search_dense(
                query_embedding=[0.0] * settings.embedding_dimension,  # Dummy embedding
                top_k=1,
                filters={"chunk_id": document_id}
            )
            
            if not doc_results:
                logger.warning(f"Document {document_id} not found")
                return []
            
            # Get the document's embedding from the result
            # Note: This is a simplified approach - in production you'd store embeddings
            source_content = doc_results[0]['content']
            source_embedding = await self.embedding_service.embed_text(source_content)
            
            # Search for similar documents
            similar_results = await self.vector_store.search_dense(
                source_embedding,
                top_k=top_k + 1  # +1 to exclude the source document
            )
            
            # Filter out the source document
            similar_results = [r for r in similar_results if r['id'] != document_id]
            
            # Convert to SearchResult objects
            search_results = []
            for result in similar_results[:top_k]:
                search_results.append(SearchResult(
                    id=result['id'],
                    content=result['content'],
                    score=result['score'],
                    metadata=result['metadata'],
                    highlights=None
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            raise