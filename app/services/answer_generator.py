import time
from typing import List, Dict, Any
from loguru import logger

from models import AskRequest, AskResponse, SourceDocument, QueryType
from services.search_engine import HybridSearchEngine
from services.embeddings import LLMService
from services.query_processor import QueryProcessor


class AnswerGenerator:
    """Orchestrates the question-answering pipeline."""
    
    def __init__(self, search_engine: HybridSearchEngine, llm_service: LLMService):
        self.search_engine = search_engine
        self.llm_service = llm_service
        self.query_processor = QueryProcessor(llm_service=llm_service)
    
    async def generate_answer(self, request: AskRequest) -> AskResponse:
        """
        Generate an answer to a natural language question.
        
        Pipeline:
        1. Process and optimize the query
        2. Retrieve relevant documents
        3. Generate answer using LLM
        4. Format response with sources
        """
        start_time = time.time()
        
        try:
            # Process the query
            processed_query = await self.query_processor.process_query(request.query)
            query_type = processed_query["query_type"]
            optimized_query = processed_query["optimized_query"]
            language = processed_query.get("language", "unknown")
            
            logger.info(f"Processed query: type={query_type}, lang={language}, optimized='{optimized_query}'")
            
            # Perform search with optimized query
            search_start = time.time()
            search_results = await self.search_engine.search(
                query=optimized_query,
                top_k=request.top_k,
                search_type="hybrid",
                rerank=True  # Enable reranking for better results
            )
            search_time = (time.time() - search_start) * 1000
            
            # Convert search results to dict format for LLM
            context_chunks = []
            for result in search_results:
                context_chunks.append({
                    'id': result.id,
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata
                })
            
            # Generate answer
            generation_start = time.time()
            answer_result = await self.llm_service.generate_answer(
                query=request.query,  # Use original query for answer generation
                context_chunks=context_chunks,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            generation_time = (time.time() - generation_start) * 1000
            
            # Prepare source documents if requested
            sources = None
            if request.include_sources and search_results:
                sources = self._prepare_sources(search_results[:5])  # Top 5 sources
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            
            return AskResponse(
                query=request.query,
                answer=answer_result["answer"],
                query_type=query_type,
                confidence_score=answer_result["confidence"],
                sources=sources,
                processing_time_ms=total_time,
                search_time_ms=search_time,
                generation_time_ms=generation_time,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            
            # Return error response
            total_time = (time.time() - start_time) * 1000
            return AskResponse(
                query=request.query,
                answer=f"I encountered an error while processing your question: {str(e)}",
                query_type=QueryType.OTHER,
                confidence_score=0.0,
                sources=None,
                processing_time_ms=total_time,
                search_time_ms=0.0,
                generation_time_ms=0.0
            )
    
    def _prepare_sources(self, search_results: List) -> List[SourceDocument]:
        """Convert search results to source documents."""
        sources = []
        
        for result in search_results:
            # Extract relevant excerpt (first 300 chars)
            excerpt = result.content[:300]
            if len(result.content) > 300:
                excerpt += "..."
            
            source = SourceDocument(
                id=result.id,
                content=excerpt,
                metadata=result.metadata,
                relevance_score=result.score
            )
            sources.append(source)
        
        return sources
    
    async def generate_multi_turn_answer(self, query: str, conversation_history: List[Dict],
                                       top_k: int = 10) -> AskResponse:
        """
        Generate answer considering conversation history.
        
        This is useful for follow-up questions.
        """
        # Build context from conversation history
        context = "Previous conversation:\n"
        for turn in conversation_history[-3:]:  # Last 3 turns
            context += f"Q: {turn.get('query', '')}\n"
            context += f"A: {turn.get('answer', '')}\n\n"
        
        # Modify current query with context
        contextualized_query = f"{context}\nCurrent question: {query}"
        
        # Create request with contextualized query
        request = AskRequest(
            query=contextualized_query,
            top_k=top_k,
            include_sources=True
        )
        
        return await self.generate_answer(request)