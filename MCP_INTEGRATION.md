# MCP (Model Context Protocol) Integration Guide

This document explains how to integrate the RAG API with n8n and other MCP-compatible clients.

## Overview

The RAG API now exposes a Model Context Protocol (MCP) server endpoint at `/mcp`. This allows AI agents and workflow automation tools like n8n to interact with your RAG system through a standardized interface.

## MCP Endpoint

- **URL**: `http://localhost:8000/mcp`
- **Method**: POST
- **Protocol**: JSON-RPC 2.0
- **Content-Type**: `application/json`

## Available MCP Tools

### 1. search_documents
Search for documents using hybrid search (semantic + keyword).

**Parameters:**
- `query` (string, required): Search query
- `top_k` (integer, optional): Number of results to return (default: 10)
- `search_type` (string, optional): "hybrid", "dense", or "sparse" (default: "hybrid")
- `rerank` (boolean, optional): Whether to rerank results using LLM (default: false)

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_documents",
    "arguments": {
      "query": "Patrick Schönfeld",
      "top_k": 5,
      "search_type": "hybrid"
    }
  },
  "id": 1
}
```

### 2. ask_question
Ask a natural language question and get an AI-generated answer with sources.

**Parameters:**
- `question` (string, required): Natural language question
- `top_k` (integer, optional): Number of source documents to retrieve (default: 5)
- `include_sources` (boolean, optional): Include source citations (default: true)

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "ask_question",
    "arguments": {
      "question": "Who is Patrick Schönfeld?",
      "top_k": 5,
      "include_sources": true
    }
  },
  "id": 2
}
```

### 3. get_collection_stats
Get statistics about the document collection.

**Parameters:** None

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_collection_stats",
    "arguments": {}
  },
  "id": 3
}
```

## n8n Integration

### Step 1: Set up MCP Client Node

1. In your n8n workflow, add an **MCP Client Tool** node
2. Configure the node:
   - **URL**: `http://your-rag-host:8000/mcp`
   - **Authentication**: None (or Bearer token if enabled)

### Step 2: List Available Tools

First, verify the connection by listing available tools:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "params": {},
  "id": 1
}
```

### Step 3: Use in AI Agent

Add the MCP Client Tool node to your AI Agent workflow:

1. Connect the MCP Client Tool to your AI Agent node
2. The agent can now use the RAG tools to:
   - Search for relevant documents
   - Answer questions based on your knowledge base
   - Get statistics about the indexed content

### Example n8n Workflow

Here's a simple workflow that uses the RAG MCP server:

1. **Trigger**: Webhook or manual trigger
2. **MCP Client Tool**: Connected to your RAG API
3. **AI Agent**: Uses the MCP tools to answer questions
4. **Response**: Returns the answer to the user

## Authentication (Optional)

If you enable authentication in your RAG API:

1. Set these environment variables:
   ```
   MCP_AUTH_ENABLED=true
   MCP_AUTH_TOKEN=your-secret-token
   ```

2. In n8n, configure the MCP Client with:
   - **Authentication**: Bearer Token
   - **Token**: Your secret token

## Testing the MCP Endpoint

You can test the MCP endpoint directly using curl:

### Initialize Connection
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {},
    "id": 1
  }'
```

### List Tools
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 1
  }'
```

### Search Documents
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_documents",
      "arguments": {
        "query": "example query",
        "top_k": 5
      }
    },
    "id": 1
  }'
```

## Resources

MCP also exposes document resources that can be listed and read:

### List Resources
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "resources/list",
    "params": {},
    "id": 1
  }'
```

### Read Resource
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "resources/read",
    "params": {
      "uri": "file://document.pdf"
    },
    "id": 1
  }'
```

## Troubleshooting

### Connection Issues
- Ensure the RAG API is running and accessible
- Check that the `/mcp` endpoint is available
- Verify CORS settings if accessing from a browser

### Authentication Errors
- Check that the Bearer token is correctly configured
- Ensure `MCP_AUTH_ENABLED` matches your setup

### Tool Execution Errors
- Check the RAG API logs for detailed error messages
- Ensure all required services (Qdrant, MinIO, Azure OpenAI) are running
- Verify that documents have been indexed before searching

## Advanced Usage

### Custom Workflows
You can create complex n8n workflows that:
- Automatically index new documents when uploaded
- Create Q&A chatbots powered by your RAG
- Build document search interfaces
- Generate reports based on document analysis

### Combining with Other n8n Nodes
The MCP Client Tool works well with:
- **HTTP Request**: Upload documents to the RAG API
- **Webhook**: Trigger searches based on external events
- **Chat**: Build conversational interfaces
- **Email**: Send search results or answers via email

## Best Practices

1. **Index Management**: Regularly refresh the index to include new documents
2. **Query Optimization**: Use specific, detailed queries for better results
3. **Result Limits**: Set appropriate `top_k` values to balance performance and relevance
4. **Reranking**: Enable reranking for important queries to improve accuracy
5. **Monitoring**: Check collection stats regularly to ensure healthy indexing