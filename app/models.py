from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(10, description="Number of results to return")
    search_type: Optional[str] = Field("hybrid", description="Search type: 'hybrid', 'dense', or 'sparse'")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class SearchResult(BaseModel):
    id: str = Field(..., description="Document chunk ID")
    content: str = Field(..., description="Text content of the chunk")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    highlights: Optional[List[str]] = Field(None, description="Highlighted matching phrases")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    search_type: str = Field(..., description="Type of search performed")


class FileUploadResponse(BaseModel):
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Detected file type")
    size_bytes: int = Field(..., description="File size in bytes")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    status: str = Field(..., description="Processing status")
    message: Optional[str] = Field(None, description="Additional status message")


class IndexingStatus(BaseModel):
    status: str = Field(..., description="Current indexing status")
    total_files: int = Field(..., description="Total files in bucket")
    indexed_files: int = Field(..., description="Number of indexed files")
    total_chunks: int = Field(..., description="Total chunks in index")
    last_updated: datetime = Field(..., description="Last update timestamp")
    errors: Optional[List[str]] = Field(None, description="Any errors encountered")


class HealthCheck(BaseModel):
    status: str = Field(..., description="Overall health status")
    services: Dict[str, bool] = Field(..., description="Individual service statuses")
    timestamp: datetime = Field(..., description="Health check timestamp")


class CollectionStats(BaseModel):
    collection_name: str = Field(..., description="Vector collection name")
    total_documents: int = Field(..., description="Total documents indexed")
    total_chunks: int = Field(..., description="Total chunks in collection")
    index_size_mb: float = Field(..., description="Index size in megabytes")
    last_updated: datetime = Field(..., description="Last update timestamp")


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    embeddings: Optional[List[float]] = Field(None, description="Dense embeddings")
    sparse_indices: Optional[List[int]] = Field(None, description="Sparse vector indices")
    sparse_values: Optional[List[float]] = Field(None, description="Sparse vector values")