# Hybrid Search RAG API

A production-ready Retrieval-Augmented Generation (RAG) system with advanced hybrid search capabilities, combining semantic understanding with keyword precision for superior document retrieval. Designed for on-premise deployment and seamless n8n integration.

## Features

- **Hybrid Search**: Combines semantic (dense) and keyword (sparse) search for optimal results
- **Advanced Ranking**: Multi-stage ranking with Reciprocal Rank Fusion and LLM-based reranking
- **Multi-Format Support**: Processes PDF, TXT, CSV, and JSON files with format-specific strategies
- **Intelligent Chunking**: Document-type specific chunking with configurable overlap
- **Azure OpenAI Integration**: Uses text-embedding-3-large (3072 dimensions) and gpt-4o-mini
- **MinIO Storage**: S3-compatible object storage with automatic deduplication
- **Qdrant Vector Database**: High-performance vector search with named vectors
- **RESTful API**: FastAPI-based endpoints with comprehensive error handling
- **MCP Server**: Model Context Protocol support for AI agent integration
- **Docker Deployment**: Fully containerized with health checks and monitoring
- **Enhanced Name Search**: Special handling for person/entity name queries

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

## Core Concepts

### Hybrid Search Architecture

This system implements a sophisticated hybrid search approach that combines the strengths of both semantic and keyword-based search:

#### 1. **Dense Vectors (Semantic Search)**
- Uses Azure OpenAI's `text-embedding-3-large` model (3072 dimensions)
- Captures semantic meaning and context
- Excellent for conceptual queries and paraphrasing
- Handles synonyms and related concepts naturally

#### 2. **Sparse Vectors (Keyword Search)**
- Uses Qdrant's built-in sparse vector implementation
- Preserves exact keyword matching capabilities
- Critical for technical terms, names, and specific identifiers
- Ensures important keywords aren't lost in semantic abstraction

### Document Ranking & Reranking Process

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Process │
    └────┬────┘
         │
    ┌────┴────┐              ┌────┴────┐
    │  Dense  │              │ Sparse  │
    │ Search  │              │ Search  │
    │(Semantic)│             │(Keyword)│
    └────┬────┘              └────┬────┘
         │                        │
         └──────────┬─────────────┘
                     │
              ┌──────┴──────┐
              │     RRF     │
              │   Fusion    │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │  Optional   │
              │ LLM Rerank  │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │   Results   │
              └─────────────┘
```

#### Stage 1: Initial Retrieval
1. **Query Processing**: User query is simultaneously:
   - Embedded into a dense vector for semantic search
   - Tokenized into sparse vectors for keyword search

2. **Parallel Search**: Both search methods run concurrently in Qdrant:
   - Dense search finds semantically similar documents
   - Sparse search finds keyword matches

3. **Reciprocal Rank Fusion (RRF)**:
   ```
   RRF_score = Σ(1 / (k + rank_i))
   ```
   - Combines results from both searches
   - `k=60` (constant) prevents bias toward top results
   - Creates unified ranking preserving both semantic and keyword relevance

#### Stage 2: LLM-Based Reranking (Optional)
1. **Context Enrichment**: Top-K results are sent to GPT-4o-mini
2. **Relevance Assessment**: LLM evaluates each result against the query
3. **Smart Reordering**: Results are reranked based on:
   - Contextual understanding
   - Query intent matching
   - Information completeness

### Key Advantages

#### 1. **Superior Retrieval Quality**
- **Best of Both Worlds**: Captures both meaning and precision
- **Robust to Query Variations**: Works with natural language and specific terms
- **Context-Aware**: Understands document relationships and intent

#### 2. **Enhanced Name Search**
- Special handling for person/entity queries
- Exact match prioritization for names
- Prevents semantic drift for proper nouns

#### 3. **Flexibility**
- **Configurable Alpha Weight** (`HYBRID_ALPHA=0.7`): Tune semantic vs keyword importance
- **Search Type Selection**: Choose hybrid, dense-only, or sparse-only per query
- **Optional Reranking**: Balance speed vs accuracy based on use case

#### 4. **Performance Optimization**
- **Parallel Processing**: Dense and sparse searches run concurrently
- **Batch Embeddings**: Efficient processing of multiple documents
- **Fallback Strategies**: Graceful degradation if one method fails

#### 5. **Document Intelligence**
- **Format-Specific Processing**: Optimal handling for PDFs, CSVs, JSON, TXT
- **Smart Chunking**: Preserves context with configurable overlap
- **Metadata Preservation**: Maintains source, position, and type information

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

### MCP (Model Context Protocol)
```bash
POST /mcp
Content-Type: application/json

# See MCP_INTEGRATION.md for detailed usage
```

### Ask
```bash
POST /ask
Content-Type: application/json

{
  "query": "your search query"
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

Key settings in `.env` that control ranking behavior:

```bash
# Chunking
CHUNK_SIZE=512          # Characters per chunk
CHUNK_OVERLAP=50        # Overlap between chunks

# Search & Ranking
HYBRID_ALPHA=0.7        # Dense vs sparse weight (0.7 = 70% semantic, 30% keyword)
TOP_K_RESULTS=10        # Final results to return
RERANK_TOP_K=20         # Candidates for LLM reranking
ENABLE_RERANKING=true   # Toggle LLM-based reranking
RRF_K=60               # Reciprocal Rank Fusion constant

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

### Search Quality Tuning

1. **Hybrid Alpha (`HYBRID_ALPHA`)**:
   - `0.0`: Pure keyword search (best for exact matches)
   - `0.5`: Balanced semantic and keyword
   - `0.7`: Default - emphasizes semantic understanding
   - `1.0`: Pure semantic search (best for concepts)

2. **Chunk Configuration**:
   - **Size**: Larger chunks (1024) preserve context, smaller (256) increase precision
   - **Overlap**: Higher overlap (100) prevents boundary loss, lower (0) maximizes coverage

3. **Reranking Strategy**:
   - Enable for critical queries requiring highest accuracy
   - Disable for real-time applications needing sub-second response
   - Adjust `RERANK_TOP_K` to balance quality vs API costs

### Performance Tips

1. **Embedding Batch Size**: Adjust `EMBEDDING_BATCH_SIZE` for API rate limits
2. **Concurrent Searches**: Hybrid search runs dense and sparse in parallel
3. **Caching**: Results are cached for repeated queries
4. **Index Optimization**: Regular index refresh maintains search quality

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

1. **Poor Semantic Results**:
   - Increase `HYBRID_ALPHA` toward 1.0
   - Check embedding model deployment
   - Verify chunk size isn't too small

2. **Missing Exact Matches**:
   - Decrease `HYBRID_ALPHA` toward 0.0
   - Ensure sparse vectors are being generated
   - Check tokenization isn't removing important terms

3. **Irrelevant Results**:
   - Enable reranking with `ENABLE_RERANKING=true`
   - Increase `RERANK_TOP_K` for more candidates
   - Adjust chunk overlap for better context

## n8n Integration

The API includes MCP (Model Context Protocol) support for seamless n8n integration:

1. Add MCP Client Tool node in n8n
2. Configure endpoint: `http://your-host:8000/mcp`
3. Use available tools:
   - `search_documents`: Search your knowledge base
   - `ask_question`: Get AI-powered answers
   - `get_collection_stats`: Monitor your RAG system

See `MCP_INTEGRATION.md` for detailed setup instructions.

## Use Cases

### Ideal For
- **Technical Documentation**: Balances technical terms with conceptual search
- **Knowledge Management**: Handles diverse query types from different users
- **Customer Support**: Finds answers using both keywords and intent
- **Research Libraries**: Combines citation search with topic exploration
- **Enterprise Search**: Handles acronyms, names, and concepts equally well

### Example Scenarios

1. **Technical Query**: "SSL certificate error"
   - Sparse search ensures "SSL" and "certificate" are found
   - Dense search includes related concepts like "TLS" or "security"

2. **Conceptual Query**: "How to improve team communication"
   - Dense search dominates, finding semantically related content
   - Sparse search still catches exact phrase matches

3. **Name Search**: "John Smith project updates"
   - Enhanced name detection prioritizes exact "John Smith" matches
   - Semantic search finds related project content

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires services running)
cd app
uvicorn main:app --reload
```

### Testing MCP Integration
```bash
# Test MCP endpoint
python test_mcp.py
```

### Adding New File Types
1. Extend `DocumentProcessor` in `services/document_processor.py`
2. Add parsing logic for the new type
3. Update chunking strategy if needed

## Enterprise Advantages

### Why Hybrid Search RAG?

1. **Accuracy**: Traditional semantic-only RAG systems can miss critical exact matches (product codes, names, technical terms). Our hybrid approach ensures nothing is lost.

2. **Flexibility**: Single embedding models can't handle all query types equally well. By combining approaches, we excel at both natural language questions and specific keyword searches.

3. **Performance**: Parallel processing and intelligent caching provide fast responses even with large document collections.

4. **Control**: On-premise deployment with configurable ranking weights gives you full control over search behavior and data security.

5. **Integration**: MCP protocol and REST API design make it easy to integrate with existing workflows, especially n8n automation.

## License

This project is provided as-is for on-premise deployment.

## Support

For issues or questions:
1. Check the logs first
2. Ensure all services are healthy
3. Verify Azure credentials are correct
4. Check example data format matches your use case
