from typing import List, Optional, Dict
from loguru import logger
import asyncio
from openai import AsyncAzureOpenAI
import tiktoken
import numpy as np

from config import settings


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        self.deployment = settings.azure_embedding_deployment
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8191  # Max tokens for text-embedding-3-large
        
    async def health_check(self) -> bool:
        """Check if the embedding service is available."""
        try:
            # Try to embed a simple test string
            test_text = "Health check"
            response = await self.client.embeddings.create(
                input=test_text,
                model=self.deployment
            )
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        try:
            # Truncate text if too long
            truncated_text = self._truncate_text(text)
            
            response = await self.client.embeddings.create(
                input=truncated_text,
                model=self.deployment
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        try:
            # Truncate texts if necessary
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            # Azure OpenAI can handle batch requests
            response = await self.client.embeddings.create(
                input=truncated_texts,
                model=self.deployment
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Fallback to individual embeddings
            embeddings = []
            for text in texts:
                try:
                    embedding = await self.embed_text(text)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to embed text: {e}")
                    # Return zero vector on failure
                    embeddings.append([0.0] * settings.embedding_dimension)
            
            return embeddings
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limits.
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:self.max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        logger.warning(f"Text truncated from {len(tokens)} to {self.max_tokens} tokens")
        return truncated_text
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Could apply query-specific preprocessing here.
        """
        # For text-embedding-3-large, queries and documents use the same method
        return await self.embed_text(query)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


class LLMService:
    """Service for LLM operations using Azure OpenAI."""
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        self.deployment = settings.azure_llm_deployment
    
    async def generate_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using LLM.
        """
        try:
            prompt = f"""Extract 5-10 important keywords from the following text. 
            Return only the keywords as a comma-separated list.
            
            Text: {text[:1000]}  # Limit text length
            
            Keywords:"""
            
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a keyword extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split(',')]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Failed to generate keywords: {e}")
            return []
    
    async def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Use LLM to rerank search results based on relevance.
        """
        try:
            # Prepare context
            context = f"Query: {query}\n\nResults:\n"
            for i, result in enumerate(results[:5]):  # Limit to top 5 for reranking
                context += f"{i+1}. {result['content'][:200]}...\n"
            
            prompt = f"""Given the search query and results below, rank the results from most to least relevant.
            Return only the numbers in order of relevance.
            
            {context}
            
            Ranking (most to least relevant):"""
            
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a search result ranking assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            ranking_text = response.choices[0].message.content.strip()
            # Parse ranking (expecting something like "3, 1, 4, 2, 5")
            try:
                ranking = [int(x.strip()) - 1 for x in ranking_text.split(',')]
                
                # Reorder results based on ranking
                reranked = []
                for idx in ranking:
                    if 0 <= idx < len(results):
                        reranked.append(results[idx])
                
                # Add any remaining results
                for i, result in enumerate(results):
                    if result not in reranked:
                        reranked.append(result)
                
                return reranked
                
            except:
                logger.warning("Failed to parse reranking, returning original order")
                return results
                
        except Exception as e:
            logger.error(f"Failed to rerank results: {e}")
            return results