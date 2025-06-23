import re
from typing import List, Dict, Any
from loguru import logger
import hashlib
import json


class ChunkingStrategy:
    """Base class for different chunking strategies."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_chunk_id(self, content: str, index: int, source: str) -> str:
        """Create a unique ID for a chunk."""
        hash_content = f"{source}_{index}_{content[:50]}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def add_metadata(self, chunk: str, index: int, source: str, 
                    additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add metadata to a chunk."""
        metadata = {
            "source": source,
            "chunk_index": index,
            "chunk_size": len(chunk),
            "chunk_id": self.create_chunk_id(chunk, index, source)
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata


class TextChunker(ChunkingStrategy):
    """Chunking strategy for plain text and unstructured documents."""
    
    def chunk(self, text: str, source: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text with overlap, trying to break at paragraph boundaries.
        """
        chunks = []
        
        # Clean text
        text = text.strip()
        if not text:
            return chunks
        
        # Split into paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk) + len(para) + 1 > self.chunk_size:
                # Save current chunk
                chunk_data = {
                    "content": current_chunk,
                    "metadata": self.add_metadata(
                        current_chunk, chunk_index, source, metadata
                    )
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                    current_chunk = " ".join(overlap_words) + "\n\n" + para
                else:
                    current_chunk = para
                    
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_data = {
                "content": current_chunk,
                "metadata": self.add_metadata(
                    current_chunk, chunk_index, source, metadata
                )
            }
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks


class StructuredDataChunker(ChunkingStrategy):
    """Chunking strategy for structured data (JSON, CSV)."""
    
    def chunk_json(self, data: Any, source: str, metadata: Dict[str, Any] = None,
                   path: str = "") -> List[Dict[str, Any]]:
        """
        Chunk JSON data by preserving structure and creating meaningful chunks.
        """
        chunks = []
        
        if isinstance(data, dict):
            # For dictionaries, create chunks for each key-value pair
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, (dict, list)):
                    # Recursively process nested structures
                    chunks.extend(self.chunk_json(value, source, metadata, current_path))
                else:
                    # Create chunk for leaf values
                    content = f"{current_path}: {json.dumps(value, ensure_ascii=False)}"
                    chunk_metadata = self.add_metadata(
                        content, len(chunks), source, metadata
                    )
                    chunk_metadata["json_path"] = current_path
                    chunk_metadata["data_type"] = "json"
                    
                    chunks.append({
                        "content": content,
                        "metadata": chunk_metadata
                    })
        
        elif isinstance(data, list):
            # For lists, create chunks for each item
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                
                if isinstance(item, (dict, list)):
                    chunks.extend(self.chunk_json(item, source, metadata, current_path))
                else:
                    content = f"{current_path}: {json.dumps(item, ensure_ascii=False)}"
                    chunk_metadata = self.add_metadata(
                        content, len(chunks), source, metadata
                    )
                    chunk_metadata["json_path"] = current_path
                    chunk_metadata["data_type"] = "json"
                    
                    chunks.append({
                        "content": content,
                        "metadata": chunk_metadata
                    })
        
        else:
            # For simple values
            content = f"{path}: {json.dumps(data, ensure_ascii=False)}" if path else str(data)
            chunk_metadata = self.add_metadata(
                content, len(chunks), source, metadata
            )
            chunk_metadata["json_path"] = path
            chunk_metadata["data_type"] = "json"
            
            chunks.append({
                "content": content,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def chunk_csv_rows(self, rows: List[Dict[str, Any]], source: str, 
                      metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk CSV data by grouping rows.
        """
        chunks = []
        chunk_index = 0
        
        # Group rows into chunks
        for i in range(0, len(rows), self.chunk_size // 100):  # Approximate rows per chunk
            chunk_rows = rows[i:i + self.chunk_size // 100]
            
            # Convert rows to readable format
            content_lines = []
            for row in chunk_rows:
                row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                content_lines.append(row_str)
            
            content = "\n".join(content_lines)
            
            chunk_metadata = self.add_metadata(
                content, chunk_index, source, metadata
            )
            chunk_metadata["row_start"] = i
            chunk_metadata["row_end"] = i + len(chunk_rows)
            chunk_metadata["data_type"] = "csv"
            
            chunks.append({
                "content": content,
                "metadata": chunk_metadata
            })
            
            chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks from CSV with {len(rows)} rows")
        return chunks


class SmartChunker:
    """
    Main chunker that selects appropriate strategy based on content type.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        self.structured_chunker = StructuredDataChunker(chunk_size, chunk_overlap)
    
    def chunk_document(self, content: Any, source: str, file_type: str,
                      metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk document based on its type.
        """
        if metadata is None:
            metadata = {}
        
        metadata["file_type"] = file_type
        
        if file_type in ["txt", "pdf"]:
            # Use text chunking for unstructured content
            if isinstance(content, str):
                return self.text_chunker.chunk(content, source, metadata)
            else:
                # If content is not string (e.g., from PDF), convert it
                text_content = str(content)
                return self.text_chunker.chunk(text_content, source, metadata)
        
        elif file_type == "json":
            # Use structured chunking for JSON
            return self.structured_chunker.chunk_json(content, source, metadata)
        
        elif file_type == "csv":
            # Use structured chunking for CSV (expecting list of dicts)
            if isinstance(content, list):
                return self.structured_chunker.chunk_csv_rows(content, source, metadata)
            else:
                # Fallback to text chunking
                return self.text_chunker.chunk(str(content), source, metadata)
        
        else:
            # Default to text chunking
            logger.warning(f"Unknown file type {file_type}, using text chunking")
            return self.text_chunker.chunk(str(content), source, metadata)