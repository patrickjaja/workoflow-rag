# Hybrid Search RAG API

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search capabilities, designed for on-premise deployment and n8n integration.

## Features

- **Hybrid Search**: Combines semantic (dense) and keyword (sparse) search for optimal results
- **Multi-Format Support**: Processes PDF, TXT, CSV, and JSON files
- **Intelligent Chunking**: Document-type specific chunking strategies
- **Azure OpenAI Integration**: Uses text-embedding-3-large for embeddings and gpt-4o-mini for reranking
- **MinIO Storage**: S3-compatible object storage for document management
- **Qdrant Vector Database**: High-performance vector search with hybrid capabilities
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Docker Deployment**: Fully containerized for easy on-premise deployment

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │────▶│   MinIO     │     │   Qdrant    │
│     API     │     │  Storage    │     │   Vector    │
└─────────────┘     └─────────────┘     │  Database   │
       │                                 └─────────────┘
       │                                        ▲
       ▼                                        │
┌─────────────┐     ┌─────────────┐            │
│ Unstructured│────▶│    Azure    │────────────┘
│  Processor  │     │   OpenAI    │
└─────────────┘     └─────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local testing)
- 8GB+ RAM recommended

### 2. Clone and Setup

```bash
git clone <repository>
cd python-rag

# Ensure .env file has your Azure credentials
# (Already configured in the provided .env)
```

### 3. Start Services

```bash
# Start all services
#docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Test the API

```bash
# Install test dependencies
pip install aiohttp

# Run test suite
python test_api.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Upload File
```bash
POST /upload
Content-Type: multipart/form-data

# Example with curl:
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
```

### Search
```bash
POST /search
Content-Type: application/json

{
  "query": "your search query",
  "top_k": 10,
  "search_type": "hybrid"  # Options: "hybrid", "dense", "sparse"
}
```

### Refresh Index
```bash
POST /index/refresh
```

### Get Statistics
```bash
GET /stats
```

## n8n Integration

This API is designed to work as a tool in n8n workflows for RAG patterns.

### n8n HTTP Request Node Configuration

1. **Search Endpoint**:
   - Method: POST
   - URL: `http://your-host:8000/search`
   - Body Type: JSON
   - Body:
     ```json
     {
       "query": "{{ $json.query }}",
       "top_k": 10,
       "search_type": "hybrid"
     }
     ```

2. **Upload Endpoint**:
   - Method: POST
   - URL: `http://your-host:8000/upload`
   - Body Type: Form-Data
   - Send Binary Data: Yes

### Example n8n Workflow

```json
{
  "nodes": [
    {
      "name": "RAG Search",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/search",
        "jsonParameters": true,
        "options": {},
        "bodyParametersJson": {
          "query": "{{ $json.userQuery }}",
          "top_k": 5
        }
      }
    }
  ]
}
```

## Configuration

Key settings in `.env`:

```bash
# Chunking
CHUNK_SIZE=512          # Characters per chunk
CHUNK_OVERLAP=50        # Overlap between chunks

# Search
HYBRID_ALPHA=0.7        # Weight for dense vs sparse (0.7 = 70% dense)
TOP_K_RESULTS=10        # Default number of results
RERANK_TOP_K=20         # Candidates for reranking

# Azure OpenAI
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_LLM_DEPLOYMENT=gpt-4o-mini
```

## Advanced Usage

### Custom Filters

Search with metadata filters:

```json
{
  "query": "workshop",
  "filters": {
    "file_type": "pdf",
    "filename": "Workshop_Manual.pdf"
  }
}
```

### Batch Processing

Process multiple files from MinIO:

```bash
# Upload files to MinIO bucket
# Then refresh index
curl -X POST http://localhost:8000/index/refresh
```

## Performance Optimization

1. **Embedding Batch Size**: Adjust `EMBEDDING_BATCH_SIZE` for GPU/API limits
2. **Chunk Size**: Larger chunks preserve more context but reduce precision
3. **Hybrid Alpha**: Tune based on your data (higher = more semantic weight)
4. **Reranking**: Disable for faster searches with `rerank=false`

## Monitoring

### Check Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
```

### MinIO Console
Access at: http://localhost:9001
- Username: minioadmin
- Password: minioadmin

### Qdrant Dashboard
Access at: http://localhost:6333/dashboard

## Troubleshooting

### Services Not Starting
```bash
# Check Docker resources
docker system df

# Restart services
docker-compose down
docker-compose up -d
```

### Slow Embeddings
- Check Azure OpenAI rate limits
- Reduce `EMBEDDING_BATCH_SIZE`
- Enable request caching

### Search Quality Issues
- Adjust `HYBRID_ALPHA` value
- Increase `CHUNK_OVERLAP`
- Enable reranking for better results

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires services running)
cd app
uvicorn main:app --reload
```

### Adding New File Types
1. Extend `DocumentProcessor` in `services/document_processor.py`
2. Add parsing logic for the new type
3. Update chunking strategy if needed

## License

This project is provided as-is for on-premise deployment.

## Support

For issues or questions:
1. Check the logs first
2. Ensure all services are healthy
3. Verify Azure credentials are correct
4. Check example data format matches your use case
