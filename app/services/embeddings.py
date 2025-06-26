from typing import List, Optional, Dict
from loguru import logger
import asyncio
from openai import AsyncAzureOpenAI, RateLimitError
import tiktoken
import numpy as np

from config import settings
from utils.retry import async_retry, RateLimiter


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
        self.rate_limiter = RateLimiter(requests_per_minute=settings.embeddings_per_minute)
        
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
    
    @async_retry(
        max_attempts=settings.max_retries,
        initial_delay=settings.initial_retry_delay,
        max_delay=settings.max_retry_delay,
        exponential_base=settings.retry_multiplier,
        exceptions=(RateLimitError,)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Truncate text if too long
            truncated_text = self._truncate_text(text)
            
            response = await self.client.embeddings.create(
                input=truncated_text,
                model=self.deployment
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except RateLimitError:
            # Re-raise rate limit errors to trigger retry
            raise
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    @async_retry(
        max_attempts=settings.max_retries,
        initial_delay=settings.initial_retry_delay,
        max_delay=settings.max_retry_delay,
        exponential_base=settings.retry_multiplier,
        exceptions=(RateLimitError,)
    )
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Truncate texts if necessary
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            # Azure OpenAI can handle batch requests
            response = await self.client.embeddings.create(
                input=truncated_texts,
                model=self.deployment
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except RateLimitError:
            # Re-raise rate limit errors to trigger retry
            raise
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # For non-rate-limit errors, fall back to individual embeddings
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    embedding = await self.embed_text(text)
                    embeddings.append(embedding)
                except Exception as individual_error:
                    logger.error(f"Failed to embed text {i}: {individual_error}")
                    # Return zero vector only as absolute last resort
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
    
    def __init__(self, config_settings=None):
        # Use provided settings or fall back to global settings
        self.settings = config_settings or settings
        self.client = AsyncAzureOpenAI(
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint
        )
        self.deployment = self.settings.azure_llm_deployment
    
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
                # Show more content and include metadata if available
                content_preview = result['content'][:500]
                if 'metadata' in result and 'filename' in result['metadata']:
                    context += f"{i+1}. [From: {result['metadata']['filename']}] {content_preview}...\n"
                else:
                    context += f"{i+1}. {content_preview}...\n"
            
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
    
    async def generate_answer(self, query: str, context_chunks: List[Dict], 
                            max_tokens: int = 500, temperature: float = 0.3) -> Dict[str, any]:
        """
        Generate a natural language answer based on retrieved chunks.
        
        Returns:
            Dict containing answer and confidence score
        """
        try:
            # Prepare context from chunks
            context = "Context information:\n\n"
            for i, chunk in enumerate(context_chunks[:5]):  # Use top 5 chunks
                context += f"[{i+1}] {chunk['content']}\n\n"
            
            # Create the prompt
            prompt = f"""Based on the context provided below, answer the following question. 
            If the answer cannot be found in the context, say so clearly.
            Be concise but comprehensive. Use only information from the context.
            
            Question: {query}
            
            {context}
            
            Answer:"""
            
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self.settings.llm_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on various factors
            confidence = self._calculate_answer_confidence(answer, context_chunks)
            
            return {
                "answer": answer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": "I encountered an error while generating the answer. Please try again.",
                "confidence": 0.0
            }
    
    def _calculate_answer_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Based on:
        - Average relevance scores of chunks
        - Whether answer indicates uncertainty
        - Length and completeness of answer
        """
        confidence = 0.0
        
        # Factor 1: Average chunk relevance scores (40% weight)
        if chunks:
            avg_score = sum(chunk.get('score', 0) for chunk in chunks) / len(chunks)
            confidence += 0.4 * min(avg_score * 2, 1.0)  # Normalize scores
        
        # Factor 2: Answer certainty (40% weight)
        uncertainty_phrases = [
            "cannot be found", "no information", "unclear", "not mentioned",
            "don't have", "unable to determine", "not available"
        ]
        answer_lower = answer.lower()
        
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            confidence += 0.1  # Low confidence if uncertain
        else:
            confidence += 0.4  # High confidence if certain
        
        # Factor 3: Answer length and quality (20% weight)
        if len(answer) > 50:  # Reasonable answer length
            confidence += 0.2
        elif len(answer) > 20:
            confidence += 0.1
        
        return round(confidence, 2)
    
    async def parse_query(self, query: str) -> Dict[str, str]:
        """
        Parse a natural language query to understand intent and extract key information.
        """
        try:
            prompt = f"""Analyze the following query and extract:
            1. The main entity or subject being asked about
            2. The type of information requested
            3. Any specific attributes mentioned
            
            Query: {query}
            
            Return in format:
            Entity: <entity>
            Information Type: <type>
            Attributes: <attributes>"""
            
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a query parsing assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            parsed = {}
            for line in result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip()] = value.strip()
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse query: {e}")
            return {}