# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hybrid Search RAG (Retrieval-Augmented Generation) API designed for on-premise deployment and n8n integration. It combines semantic and keyword search using Qdrant vector database, MinIO object storage, and Azure OpenAI for embeddings.

## Key Architecture Components

### Service Dependencies
- **Qdrant**: Vector database for hybrid search (dense + sparse vectors)
- **MinIO**: S3-compatible object storage for documents
- **Azure OpenAI**: Embeddings (text-embedding-3-large) and LLM operations (gpt-4o-mini)
- **FastAPI**: REST API framework

### Core Search Flow
1. Documents uploaded to MinIO → Processed by Unstructured library → Chunked based on file type
2. Chunks embedded using Azure OpenAI → Stored in Qdrant with both dense and sparse vectors
3. Search queries use hybrid approach: semantic (dense) + keyword (sparse) with configurable weights
4. Results can be reranked using LLM for improved relevance

## Common Development Commands

### Docker Operations
```bash
# Build and start all services
docker-compose build app
docker-compose up -d

# View logs
docker-compose logs -f app
docker-compose logs -f qdrant
docker-compose logs -f minio

# Restart specific service
docker-compose restart app

# Full reset
docker-compose down -v
docker-compose up -d
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI locally (requires Docker services running)
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python test_api.py
```

### Testing API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Upload file
curl -X POST -F "file=@example_data/document.pdf" http://localhost:8000/upload

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 10, "search_type": "hybrid"}'

# Refresh index from MinIO
curl -X POST http://localhost:8000/index/refresh
```

## Known Issues & Solutions

### Container Import Errors
- **ModuleNotFoundError**: Often due to missing dependencies in requirements.txt or import path issues
- **Unstructured library compatibility**: Use version 0.10.30 with pdfminer.six==20231228
- **Python path**: Dockerfile sets `ENV PYTHONPATH=/app` to resolve import issues

### Azure OpenAI Configuration
- Credentials are in `.env` file
- Rate limiting: Adjust `EMBEDDING_BATCH_SIZE` if hitting limits
- Embedding dimension: 3072 for text-embedding-3-large

### Search Quality Tuning
- `HYBRID_ALPHA=0.7`: Controls dense vs sparse weight (higher = more semantic)
- `CHUNK_SIZE=512`: Balance between context and precision
- `CHUNK_OVERLAP=50`: Prevents information loss at chunk boundaries

## Key Implementation Details

### Hybrid Search Strategy (app/services/vector_store.py)
- Uses Qdrant's named vectors: "dense" for embeddings, "sparse" for keywords
- Implements Reciprocal Rank Fusion (RRF) for result combination
- Fallback to dense search if hybrid fails

### Document Processing (app/services/document_processor.py)
- PDF: Uses Unstructured with "hi_res" strategy
- CSV: Converts to list of dictionaries, chunks by rows
- JSON: Preserves structure, creates chunks for each path
- TXT: Chunks at paragraph boundaries with overlap

### Chunking Logic (app/utils/chunking.py)
- Document-type specific strategies
- Metadata preservation (source, chunk_index, file_type)
- Smart chunking for structured data maintains context

## n8n Integration Notes

The API is designed as a RAG tool for n8n workflows:
- All endpoints return JSON responses
- Search endpoint accepts filters for metadata-based filtering
- Upload endpoint handles multipart form data
- Batch processing supported via index refresh

## Environment Variables

Critical settings in `.env`:
- Azure OpenAI credentials (endpoint, API key, deployments)
- MinIO credentials (default: minioadmin/minioadmin)
- Qdrant connection settings
- Search parameters (chunk size, overlap, alpha weight)