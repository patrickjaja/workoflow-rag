"""
MCP (Model Context Protocol) Server implementation for RAG API.
Provides standardized interface for AI agents to interact with the RAG system.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import asyncio
from loguru import logger
from pydantic import BaseModel, Field

from services.search_engine import HybridSearchEngine
from services.answer_generator import AnswerGenerator
from services.vector_store import VectorStore
from services.minio_client import MinIOClient
from services.document_processor import DocumentProcessor
from models import SearchRequest, AskRequest


class MCPRequest(BaseModel):
    """MCP JSON-RPC 2.0 Request"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Union[str, int, None] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC 2.0 Response"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Union[str, int, None] = None


class MCPError:
    """Standard MCP error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    

class MCPServer:
    """
    MCP Server implementation that exposes RAG capabilities as tools and resources.
    """
    
    def __init__(
        self,
        search_engine: HybridSearchEngine,
        answer_generator: AnswerGenerator,
        vector_store: VectorStore,
        minio_client: MinIOClient,
        document_processor: DocumentProcessor
    ):
        self.search_engine = search_engine
        self.answer_generator = answer_generator
        self.vector_store = vector_store
        self.minio_client = minio_client
        self.document_processor = document_processor
        
        # Server metadata
        self.server_info = {
            "name": self.settings.mcp_server_name,
            "version": self.settings.mcp_server_version,
            "vendor": self.settings.mcp_server_vendor
        }
        
        # Available methods mapping
        self.methods = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "completion/complete": self._handle_completion
        }
        
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for handling MCP requests.
        """
        try:
            # Parse request
            request = MCPRequest(**request_data)
            
            # Check if method exists
            if request.method not in self.methods:
                return self._create_error_response(
                    MCPError.METHOD_NOT_FOUND,
                    f"Method '{request.method}' not found",
                    request.id
                )
            
            # Execute method
            handler = self.methods[request.method]
            result = await handler(request.params or {})
            
            # Return success response
            return MCPResponse(
                result=result,
                id=request.id
            ).model_dump(exclude_none=True)
            
        except Exception as e:
            logger.error(f"MCP request handling error: {e}")
            return self._create_error_response(
                MCPError.INTERNAL_ERROR,
                str(e),
                request_data.get("id") if isinstance(request_data, dict) else None
            )
    
    def _create_error_response(self, code: int, message: str, request_id: Any) -> Dict[str, Any]:
        """Create standard MCP error response."""
        return MCPResponse(
            error={
                "code": code,
                "message": message
            },
            id=request_id
        ).model_dump(exclude_none=True)
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialization request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "completion": {}
            },
            "serverInfo": self.server_info
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools."""
        return {
            "tools": [
                {
                    "name": "search_documents",
                    "description": "Search for documents using hybrid search (semantic + keyword)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 10
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["hybrid", "dense", "sparse"],
                                "description": "Type of search to perform",
                                "default": "hybrid"
                            },
                            "rerank": {
                                "type": "boolean",
                                "description": "Whether to rerank results using LLM",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "ask_question",
                    "description": "Ask a natural language question and get an AI-generated answer with sources",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of source documents to retrieve",
                                "default": 5
                            },
                            "include_sources": {
                                "type": "boolean",
                                "description": "Include source citations in response",
                                "default": True
                            }
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
        }
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "search_documents":
                # Create search request
                search_request = SearchRequest(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 10),
                    search_type=arguments.get("search_type", "hybrid"),
                    rerank=arguments.get("rerank", False)
                )
                
                # Perform search
                results = await self.search_engine.search(
                    query=search_request.query,
                    top_k=search_request.top_k,
                    search_type=search_request.search_type,
                    rerank=search_request.rerank
                )
                
                # Format results for MCP
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "results": [
                                    {
                                        "content": r.content,
                                        "metadata": r.metadata,
                                        "score": r.score
                                    } for r in results
                                ],
                                "count": len(results)
                            }, indent=2)
                        }
                    ]
                }
                
            elif tool_name == "ask_question":
                # Create ask request
                ask_request = AskRequest(
                    question=arguments["question"],
                    top_k=arguments.get("top_k", 5),
                    include_sources=arguments.get("include_sources", True)
                )
                
                # Generate answer
                response = await self.answer_generator.generate_answer(ask_request)
                
                # Format response for MCP
                result_data = {
                    "answer": response.answer,
                    "confidence": response.confidence_score
                }
                
                if response.sources:
                    result_data["sources"] = [
                        {
                            "content": s.content,
                            "metadata": s.metadata,
                            "relevance": s.relevance_score
                        } for s in response.sources
                    ]
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result_data, indent=2)
                        }
                    ]
                }
                
            elif tool_name == "get_collection_stats":
                # Get collection statistics
                stats = await self.vector_store.get_collection_stats()
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "total_documents": stats.total_documents,
                                "total_chunks": stats.total_chunks,
                                "collection_name": stats.collection_name,
                                "index_size_mb": stats.index_size_mb,
                                "last_updated": stats.last_updated.isoformat() if stats.last_updated else None
                            }, indent=2)
                        }
                    ]
                }
                
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool '{tool_name}': {str(e)}"
                    }
                ],
                "isError": True
            }
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available resources."""
        try:
            # Get list of files from MinIO
            files = await self.minio_client.list_files()
            
            resources = []
            for file_info in files[:20]:  # Limit to first 20 files
                resources.append({
                    "uri": f"file://{file_info['name']}",
                    "name": file_info['name'],
                    "description": f"Document: {file_info['name']} (Size: {file_info['size']} bytes)",
                    "mimeType": self._get_mime_type(file_info['name'])
                })
            
            return {"resources": resources}
            
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            return {"resources": []}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a specific resource."""
        uri = params.get("uri", "")
        
        if not uri.startswith("file://"):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Invalid resource URI format"
                    }
                ],
                "isError": True
            }
        
        filename = uri.replace("file://", "")
        
        try:
            # Get file metadata from vector store
            stats = await self.vector_store.get_collection_stats()
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "filename": filename,
                            "status": "indexed",
                            "total_chunks": stats.total_chunks,
                            "message": f"Document '{filename}' is indexed and searchable"
                        }, indent=2)
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error reading resource: {str(e)}"
                    }
                ],
                "isError": True
            }
    
    async def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle completion requests for better UX."""
        ref = params.get("ref", {})
        
        # Provide completion based on context
        if ref.get("type") == "resource":
            # Suggest available files
            try:
                files = await self.minio_client.list_files()
                return {
                    "completion": {
                        "values": [f"file://{f['name']}" for f in files[:10]]
                    }
                }
            except:
                return {"completion": {"values": []}}
        
        return {"completion": {"values": []}}
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type based on file extension."""
        extension = filename.split('.')[-1].lower() if '.' in filename else ''
        mime_map = {
            'pdf': 'application/pdf',
            'txt': 'text/plain',
            'csv': 'text/csv',
            'json': 'application/json'
        }
        return mime_map.get(extension, 'application/octet-stream')