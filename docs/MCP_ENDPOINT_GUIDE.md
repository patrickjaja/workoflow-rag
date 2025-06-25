# MCP Endpoint Configuration Guide

This guide provides comprehensive documentation for the Model Context Protocol (MCP) endpoint in the Python RAG API.

## Overview

The `/mcp` endpoint implements the JSON-RPC 2.0 based Model Context Protocol, allowing AI agents and automation tools (like n8n, Claude, etc.) to interact with the RAG system through a standardized interface.

## Endpoint Details

- **URL**: `POST http://localhost:8000/mcp`
- **Protocol**: JSON-RPC 2.0
- **Content-Type**: `application/json`
- **Authentication**: Optional Bearer token

## Authentication Setup

### 1. Enable Authentication

To enable authentication for the MCP endpoint, update your `.env` file:

```bash
# Enable MCP authentication
MCP_AUTH_ENABLED=true

# Set your secure token
MCP_AUTH_TOKEN=your-secure-token-here
```

### 2. Generate a Secure Token

```bash
# Generate a secure random token
openssl rand -hex 32
```

### 3. Using Authentication in Requests

When authentication is enabled, include the Bearer token in your requests:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secure-token-here" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}'
```

## Available MCP Methods

### 1. Initialize Connection

Establishes an MCP session and returns server capabilities.

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

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "0.1.0",
    "capabilities": {
      "tools": {
        "available": true
      },
      "resources": {
        "available": true,
        "subscriptions": false
      }
    },
    "serverInfo": {
      "name": "rag-mcp-server",
      "version": "1.0.0",
      "vendor": "workoflow-rag"
    }
  },
  "id": 1
}
```

### 2. List Available Tools

Get a list of all available tools in the RAG system.

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
  }'
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "search_documents",
        "description": "Search through indexed documents using hybrid search",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 10},
            "search_type": {"type": "string", "enum": ["hybrid", "dense", "sparse"], "default": "hybrid"}
          },
          "required": ["query"]
        }
      },
      {
        "name": "ask_question",
        "description": "Ask a natural language question and get an AI-generated answer",
        "inputSchema": {
          "type": "object",
          "properties": {
            "question": {"type": "string", "description": "The question to ask"},
            "top_k": {"type": "integer", "default": 10},
            "include_sources": {"type": "boolean", "default": true}
          },
          "required": ["question"]
        }
      },
      {
        "name": "get_collection_stats",
        "description": "Get statistics about the document collection",
        "inputSchema": {
          "type": "object",
          "properties": {}
        }
      }
    ]
  },
  "id": 2
}
```

### 3. Search Documents

Search through indexed documents using the hybrid search capability.

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_documents",
      "arguments": {
        "query": "employee benefits policy",
        "top_k": 5,
        "search_type": "hybrid"
      }
    },
    "id": 3
  }'
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\": [...], \"count\": 5, \"search_type\": \"hybrid\"}"
      }
    ]
  },
  "id": 3
}
```

### 4. Ask a Question

Get an AI-generated answer to a natural language question.

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "ask_question",
      "arguments": {
        "question": "What is the company policy on remote work?",
        "top_k": 10,
        "include_sources": true
      }
    },
    "id": 4
  }'
```

**Response Example:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"answer\": \"Based on the documents...\", \"confidence\": 0.85, \"sources\": [...]}"
      }
    ]
  },
  "id": 4
}
```

### 5. Get Collection Statistics

Retrieve statistics about the document collection.

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_collection_stats",
      "arguments": {}
    },
    "id": 5
  }'
```

### 6. List Resources

Get a list of all indexed documents.

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "resources/list",
    "params": {},
    "id": 6
  }'
```

### 7. Read a Resource

Read the content of a specific document.

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "resources/read",
    "params": {
      "uri": "file://documents/example.pdf"
    },
    "id": 7
  }'
```

## Error Handling

The MCP endpoint returns standard JSON-RPC 2.0 error responses:

### Authentication Error (HTTP 401)
```json
{
  "error": "Missing Bearer token",
  "detail": "Authentication required"
}
```

### Method Not Found
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method 'invalid/method' not found"
  },
  "id": 8
}
```

### Invalid Parameters
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid params: Missing required parameter 'query'"
  },
  "id": 9
}
```

### Internal Error
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32603,
    "message": "Internal error: Database connection failed"
  },
  "id": 10
}
```

## n8n Integration Example

### 1. HTTP Request Node Configuration

```json
{
  "method": "POST",
  "url": "http://your-rag-api:8000/mcp",
  "authentication": "genericCredentialType",
  "genericAuthType": "httpHeaderAuth",
  "sendHeaders": true,
  "headerParameters": {
    "parameters": [
      {
        "name": "Content-Type",
        "value": "application/json"
      },
      {
        "name": "Authorization",
        "value": "Bearer {{$credentials.mcpToken}}"
      }
    ]
  },
  "sendBody": true,
  "bodyParameters": {
    "parameters": [
      {
        "name": "jsonrpc",
        "value": "2.0"
      },
      {
        "name": "method",
        "value": "tools/call"
      },
      {
        "name": "params",
        "value": {
          "name": "ask_question",
          "arguments": {
            "question": "={{$json.question}}",
            "top_k": 10,
            "include_sources": true
          }
        }
      },
      {
        "name": "id",
        "value": "={{$json.requestId}}"
      }
    ]
  }
}
```

### 2. Workflow Example: Document Q&A Bot

1. **Webhook Trigger**: Receive questions from Slack/Teams
2. **HTTP Request**: Call MCP endpoint with `ask_question` tool
3. **JSON Parse**: Extract answer from response
4. **Send Response**: Reply back to the user

### 3. Workflow Example: Document Search API

1. **HTTP Trigger**: REST endpoint for search
2. **HTTP Request**: Call MCP endpoint with `search_documents` tool
3. **Transform Data**: Format results for frontend
4. **Respond to Webhook**: Return search results

## Testing the MCP Endpoint

Use the provided test script to validate your MCP implementation:

```bash
# Run the comprehensive MCP test suite
python test_mcp.py

# Test with authentication enabled
MCP_AUTH_ENABLED=true MCP_AUTH_TOKEN=test-token python test_mcp.py
```

## Monitoring and Debugging

### 1. Enable Debug Logging

Set in your `.env` file:
```bash
LOG_LEVEL=DEBUG
```

### 2. Monitor MCP Requests

View logs for MCP activity:
```bash
docker-compose logs -f app | grep -E "(MCP|mcp)"
```

### 3. Common Issues

1. **Authentication Failures**
   - Check if `MCP_AUTH_ENABLED` matches your client configuration
   - Verify the Bearer token format and value

2. **Method Not Found**
   - Ensure you're using the correct method name
   - Check the `tools/list` response for available methods

3. **Invalid Parameters**
   - Verify parameter names match the input schema
   - Check data types (strings, integers, booleans)

4. **Connection Issues**
   - Ensure the API is accessible from your client
   - Check firewall and network settings
   - Verify the correct port (default: 8000)

## Security Best Practices

1. **Always use authentication in production**
   ```bash
   MCP_AUTH_ENABLED=true
   MCP_AUTH_TOKEN=$(openssl rand -hex 32)
   ```

2. **Use HTTPS in production**
   - Deploy behind a reverse proxy (nginx, traefik)
   - Configure SSL certificates

3. **Implement rate limiting**
   - Use nginx rate limiting
   - Add application-level rate limiting

4. **Monitor access logs**
   - Track MCP usage patterns
   - Alert on suspicious activity

5. **Rotate tokens regularly**
   - Update tokens monthly
   - Use secret management tools

## Performance Optimization

1. **Batch Operations**
   - Use multiple tool calls in a single request when possible
   - Minimize round trips

2. **Caching**
   - Cache frequently asked questions
   - Use Redis for session management

3. **Connection Pooling**
   - Reuse HTTP connections
   - Configure appropriate timeouts

4. **Resource Limits**
   - Set appropriate `top_k` values
   - Limit response sizes for large documents