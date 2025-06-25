from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
import sys
from datetime import datetime
import time
import os
from typing import List

from config import settings
from models import (
    SearchRequest, SearchResponse, FileUploadResponse,
    IndexingStatus, HealthCheck, CollectionStats,
    AskRequest, AskResponse, MCPRequest, MCPResponse
)
from services.vector_store import VectorStore
from services.minio_client import MinIOClient
from services.embeddings import EmbeddingService, LLMService
from services.document_processor import DocumentProcessor
from services.search_engine import HybridSearchEngine
from services.answer_generator import AnswerGenerator
from services.mcp_server import MCPServer


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)


# Global instances
vector_store = None
minio_client = None
embedding_service = None
document_processor = None
search_engine = None
llm_service = None
answer_generator = None
mcp_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_store, minio_client, embedding_service, document_processor, search_engine, llm_service, answer_generator, mcp_server
    
    logger.info("Initializing services...")
    
    try:
        # Initialize services
        vector_store = VectorStore()
        await vector_store.initialize()
        
        minio_client = MinIOClient()
        await minio_client.initialize()
        
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        document_processor = DocumentProcessor(embedding_service)
        search_engine = HybridSearchEngine(vector_store, embedding_service)
        answer_generator = AnswerGenerator(search_engine, llm_service)
        
        # Initialize MCP server
        mcp_server = MCPServer(
            search_engine=search_engine,
            answer_generator=answer_generator,
            vector_store=vector_store,
            minio_client=minio_client,
            document_processor=document_processor,
            settings=settings
        )
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint returning API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": f"{settings.app_name} for semantic and keyword search",
        "endpoints": {
            "search": "/search",
            "upload": "/upload",
            "index": "/index/refresh",
            "health": "/health",
            "stats": "/stats",
            "mcp": "/mcp"
        }
    }


@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check():
    """Check health status of all services."""
    services_status = {
        "fastapi": True,
        "qdrant": False,
        "minio": False,
        "azure_openai": False
    }
    
    try:
        # Check Qdrant
        if vector_store:
            await vector_store.health_check()
            services_status["qdrant"] = True
    except:
        pass
    
    try:
        # Check MinIO
        if minio_client:
            await minio_client.health_check()
            services_status["minio"] = True
    except:
        pass
    
    try:
        # Check Azure OpenAI
        if embedding_service:
            await embedding_service.health_check()
            services_status["azure_openai"] = True
    except:
        pass
    
    overall_status = "healthy" if all(services_status.values()) else "unhealthy"
    
    return HealthCheck(
        status=overall_status,
        services=services_status,
        timestamp=datetime.utcnow()
    )


@app.get("/stats", response_model=CollectionStats, tags=["General"])
async def get_stats():
    """Get collection statistics."""
    try:
        stats = await vector_store.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Perform hybrid search on indexed documents.
    
    Search types:
    - hybrid: Combines semantic and keyword search (default)
    - dense: Semantic search only
    - sparse: Keyword search only
    """
    try:
        start_time = time.time()
        
        results = await search_engine.search(
            query=request.query,
            top_k=request.top_k,
            search_type=request.search_type,
            filters=request.filters,
            rerank=request.rerank
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            search_type=request.search_type
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    """
    Ask a natural language question and get an AI-generated answer.
    
    This endpoint:
    - Understands natural language questions
    - Retrieves relevant information from indexed documents
    - Generates a comprehensive answer using LLM
    - Provides source attribution for transparency
    
    Example questions:
    - "Who is Patrick Schönfeld?"
    - "What is the role of John Smith in the company?"
    - "Where is the Munich office located?"
    """
    try:
        # Generate answer using the answer generator
        response = await answer_generator.generate_answer(request)
        return response
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=FileUploadResponse, tags=["Upload"])
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a file for indexing.
    
    Supported file types: PDF, TXT, CSV, JSON
    """
    try:
        start_time = time.time()
        
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['pdf', 'txt', 'csv', 'json']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Save file temporarily
        temp_path = f"/tmp/uploads/{file.filename}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Upload to MinIO
        file_id = await minio_client.upload_file(temp_path, file.filename)
        
        # Delete existing chunks for this filename before indexing new ones
        deleted_count = await vector_store.delete_by_filename(file.filename)
        if deleted_count > 0:
            logger.info(f"Replaced {deleted_count} existing chunks for {file.filename}")
        
        # Process document
        chunks = await document_processor.process_file(temp_path, file.filename)
        
        # Index chunks
        await vector_store.add_documents(chunks)
        
        # Clean up temp file
        os.remove(temp_path)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        message = "File indexed successfully"
        if deleted_count > 0:
            message = f"File indexed successfully (replaced {deleted_count} existing chunks)"
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_type=file_extension,
            size_bytes=len(content),
            chunks_created=len(chunks),
            processing_time_ms=processing_time_ms,
            status="success",
            message=message
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/refresh", response_model=IndexingStatus, tags=["Index"])
async def refresh_index(background_tasks: BackgroundTasks):
    """
    Refresh the index by processing all files in MinIO bucket.
    """
    try:
        # Get list of files from MinIO
        files = await minio_client.list_files()
        
        indexed_count = 0
        errors = []
        total_chunks = 0
        
        for file_info in files:
            try:
                # Download file
                temp_path = f"/tmp/uploads/{file_info['name']}"
                await minio_client.download_file(file_info['name'], temp_path)
                
                # Delete existing chunks for this file before re-indexing
                deleted_count = await vector_store.delete_by_filename(file_info['name'])
                if deleted_count > 0:
                    logger.info(f"Replacing {deleted_count} existing chunks for {file_info['name']}")
                
                # Process and index
                chunks = await document_processor.process_file(temp_path, file_info['name'])
                await vector_store.add_documents(chunks)
                
                indexed_count += 1
                total_chunks += len(chunks)
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                errors.append(f"Failed to process {file_info['name']}: {str(e)}")
                logger.error(f"Failed to process {file_info['name']}: {e}")
        
        return IndexingStatus(
            status="completed",
            total_files=len(files),
            indexed_files=indexed_count,
            total_chunks=total_chunks,
            last_updated=datetime.utcnow(),
            errors=errors if errors else None
        )
        
    except Exception as e:
        logger.error(f"Index refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp", tags=["MCP"])
async def mcp_endpoint(request: Request):
    """
    Model Context Protocol (MCP) endpoint for AI agent integration.
    
    This endpoint implements the MCP JSON-RPC 2.0 protocol, allowing AI agents
    (like n8n, Claude, etc.) to interact with the RAG system through a standardized interface.
    
    Available methods:
    - initialize: Initialize MCP connection
    - tools/list: List available tools
    - tools/call: Execute a tool
    - resources/list: List available resources
    - resources/read: Read a specific resource
    """
    try:
        # Check authentication if enabled
        if settings.mcp_auth_enabled:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing Bearer token")
            
            token = auth_header.replace("Bearer ", "")
            if token != settings.mcp_auth_token:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Get request body
        request_data = await request.json()
        
        # Handle request through MCP server
        response = await mcp_server.handle_request(request_data)
        
        # Return JSON-RPC response
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP endpoint error: {e}")
        # Return MCP-compliant error response
        error_response = MCPResponse(
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            id=request_data.get("id") if "request_data" in locals() else None
        )
        return JSONResponse(
            content=error_response.model_dump(exclude_none=True),
            status_code=200  # JSON-RPC errors still return 200 HTTP status
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)