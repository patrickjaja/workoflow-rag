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
    
    def json_to_readable_text(self, obj: Any, indent: int = 0) -> str:
        """
        Convert JSON object to human-readable text format optimized for LLMs.
        Works dynamically with any JSON structure.
        """
        indent_str = "  " * indent
        
        if isinstance(obj, dict):
            lines = []
            for key, value in obj.items():
                # Format key nicely (replace underscores with spaces, capitalize)
                formatted_key = key.replace('_', ' ').title()
                
                if isinstance(value, dict):
                    lines.append(f"{indent_str}{formatted_key}:")
                    lines.append(self.json_to_readable_text(value, indent + 1))
                elif isinstance(value, list):
                    if not value:  # Empty list
                        lines.append(f"{indent_str}{formatted_key}: []")
                    elif all(isinstance(item, (str, int, float, bool)) for item in value):
                        # Simple list - show inline
                        lines.append(f"{indent_str}{formatted_key}: {', '.join(str(v) for v in value)}")
                    else:
                        # Complex list - show each item on new line
                        lines.append(f"{indent_str}{formatted_key}:")
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                lines.append(self.json_to_readable_text(item, indent + 1))
                                if i < len(value) - 1:
                                    lines.append("")  # Empty line between complex items
                            else:
                                lines.append(f"{indent_str}  - {item}")
                elif value is None:
                    continue  # Skip None values
                elif value == "":
                    continue  # Skip empty strings
                else:
                    lines.append(f"{indent_str}{formatted_key}: {value}")
            
            return "\n".join(lines)
        
        elif isinstance(obj, list):
            lines = []
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    lines.append(self.json_to_readable_text(item, indent))
                    if i < len(obj) - 1:
                        lines.append("")  # Empty line between items
                else:
                    lines.append(f"{indent_str}- {item}")
            return "\n".join(lines)
        
        else:
            return f"{indent_str}{obj}"
    
    def chunk_json(self, data: Any, source: str, metadata: Dict[str, Any] = None,
                   path: str = "") -> List[Dict[str, Any]]:
        """
        Chunk JSON data by preserving structure and creating meaningful chunks.
        For root-level arrays, each element becomes a single chunk.
        """
        chunks = []
        
        # Special handling for root-level arrays (like employee data)
        if path == "" and isinstance(data, list):
            logger.info(f"Processing root-level JSON array with {len(data)} items")
            
            for i, item in enumerate(data):
                # Convert JSON object to readable text
                readable_content = self.json_to_readable_text(item)
                
                chunk_metadata = self.add_metadata(
                    readable_content, i, source, metadata
                )
                chunk_metadata["json_path"] = f"[{i}]"
                chunk_metadata["data_type"] = "json"
                chunk_metadata["is_complete_record"] = True
                
                # If the item has an ID, add it to metadata
                if isinstance(item, dict) and "id" in item:
                    chunk_metadata["record_id"] = item["id"]
                
                # If the item has a name, add it to metadata for easier reference
                if isinstance(item, dict) and "name" in item:
                    chunk_metadata["record_name"] = item["name"]
                
                chunks.append({
                    "content": readable_content,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunks)} chunks from JSON array (one per record)")
            return chunks
        
        # Original behavior for non-array or nested JSON
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
            # For lists (non-root), create chunks for each item
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
        Chunk CSV data - one chunk per row.
        """
        chunks = []
        
        # Create one chunk per row
        for i, row in enumerate(rows):
            # Convert row to readable format
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            
            chunk_metadata = self.add_metadata(
                row_str, i, source, metadata
            )
            chunk_metadata["row_start"] = i
            chunk_metadata["row_end"] = i  # Same as row_start since it's a single row
            chunk_metadata["data_type"] = "csv"
            
            chunks.append({
                "content": row_str,
                "metadata": chunk_metadata
            })
        
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