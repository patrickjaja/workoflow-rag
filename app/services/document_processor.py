import os
from typing import List, Dict, Any
from loguru import logger
import json
import pandas as pd
from datetime import datetime
import asyncio

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.csv import partition_csv
from unstructured.partition.json import partition_json
from unstructured.cleaners.core import clean, clean_extra_whitespace

from config import settings
from utils.chunking import SmartChunker
from models import DocumentChunk


class DocumentProcessor:
    """Process various document types using Unstructured library."""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.chunker = SmartChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    async def process_file(self, file_path: str, filename: str) -> List[DocumentChunk]:
        """
        Process a file and return chunks with embeddings.
        """
        logger.info(f"Processing file: {filename}")
        
        try:
            # Detect file type
            file_extension = filename.split('.')[-1].lower()
            
            # Extract content based on file type
            if file_extension == 'pdf':
                content = await self._process_pdf(file_path)
            elif file_extension == 'txt':
                content = await self._process_text(file_path)
            elif file_extension == 'csv':
                content = await self._process_csv(file_path)
            elif file_extension == 'json':
                content = await self._process_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create metadata
            base_metadata = {
                "filename": filename,
                "file_path": file_path,
                "processed_at": datetime.utcnow().isoformat(),
                "file_type": file_extension
            }
            
            # Chunk the content
            chunks = self.chunker.chunk_document(
                content=content,
                source=filename,
                file_type=file_extension,
                metadata=base_metadata
            )
            
            # Create embeddings for chunks
            document_chunks = await self._create_embeddings(chunks)
            
            logger.info(f"Successfully processed {filename}: {len(document_chunks)} chunks created")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
    
    async def _process_pdf(self, file_path: str) -> str:
        """Process PDF file using Unstructured."""
        try:
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",  # Use high resolution for better extraction
                infer_table_structure=True,
                include_page_breaks=True
            )
            
            # Combine elements into text
            text_content = []
            for element in elements:
                # Clean the text
                cleaned_text = clean_extra_whitespace(str(element))
                cleaned_text = clean(cleaned_text, bullets=True)
                
                if cleaned_text.strip():
                    text_content.append(cleaned_text)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            # Fallback to basic processing
            elements = partition(file_path)
            return "\n\n".join([str(el) for el in elements])
    
    async def _process_text(self, file_path: str) -> str:
        """Process text file."""
        try:
            elements = partition_text(filename=file_path)
            
            text_content = []
            for element in elements:
                cleaned_text = clean_extra_whitespace(str(element))
                if cleaned_text.strip():
                    text_content.append(cleaned_text)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            # Fallback to direct reading
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    async def _process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file."""
        try:
            # Use pandas for better CSV handling
            df = pd.read_csv(file_path)
            
            # Convert to list of dictionaries
            records = df.to_dict('records')
            
            # Clean up the records
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    # Handle NaN values
                    if pd.isna(value):
                        cleaned_record[key] = ""
                    else:
                        cleaned_record[key] = str(value)
                cleaned_records.append(cleaned_record)
            
            return cleaned_records
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            # Fallback to unstructured
            elements = partition_csv(filename=file_path)
            return [{"content": str(el)} for el in elements]
    
    async def _process_json(self, file_path: str) -> Any:
        """Process JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
            
        except Exception as e:
            logger.error(f"Error processing JSON: {e}")
            # Fallback to unstructured
            elements = partition_json(filename=file_path)
            return {"content": "\n".join([str(el) for el in elements])}
    
    async def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Create embeddings for chunks."""
        document_chunks = []
        
        # Extract texts for batch embedding
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        batch_size = settings.embedding_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = await self.embedding_service.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Add a small delay between batches to avoid overwhelming the API
            if i + batch_size < len(texts):
                await asyncio.sleep(0.5)
        
        # Create DocumentChunk objects
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            # Generate sparse vectors (simplified BM25-like approach)
            sparse_indices, sparse_values = self._generate_sparse_vectors(chunk["content"])
            
            doc_chunk = DocumentChunk(
                id=chunk["metadata"]["chunk_id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                embeddings=embedding,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks
    
    def _generate_sparse_vectors(self, text: str) -> tuple[List[int], List[float]]:
        """
        Generate simplified sparse vectors for keyword search.
        In production, you might want to use BM25 or SPLADE.
        """
        # Simple tokenization and counting
        words = text.lower().split()
        word_counts = {}
        
        for word in words:
            # Simple hash to get index
            index = abs(hash(word)) % 10000  # Limit to 10k dimensions
            word_counts[index] = word_counts.get(index, 0) + 1
        
        # Convert to sparse format
        indices = list(word_counts.keys())
        values = [float(count) for count in word_counts.values()]
        
        # Normalize
        max_val = max(values) if values else 1.0
        values = [v / max_val for v in values]
        
        return indices, values