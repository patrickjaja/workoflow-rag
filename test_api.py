#!/usr/bin/env python3
"""
Test script for the Hybrid Search RAG API.
Tests file upload, indexing, and search functionality.
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path
import time


API_BASE_URL = "http://localhost:8000"
EXAMPLE_DATA_DIR = Path("example_data")


async def test_health_check(session):
    """Test health check endpoint."""
    print("\n1. Testing health check...")
    async with session.get(f"{API_BASE_URL}/health") as resp:
        if resp.status == 200:
            data = await resp.json()
            print(f"✓ Health check passed: {data['status']}")
            print(f"  Services: {json.dumps(data['services'], indent=2)}")
            return data['services']
        else:
            print(f"✗ Health check failed: {resp.status}")
            return None


async def test_upload_file(session, file_path):
    """Test file upload."""
    print(f"\n2. Uploading file: {file_path.name}")
    
    with open(file_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename=file_path.name)
        
        async with session.post(f"{API_BASE_URL}/upload", data=data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✓ File uploaded successfully:")
                print(f"  - File ID: {result['file_id']}")
                print(f"  - Chunks created: {result['chunks_created']}")
                print(f"  - Processing time: {result['processing_time_ms']:.2f}ms")
                return True
            else:
                error = await resp.text()
                print(f"✗ Upload failed: {resp.status} - {error}")
                return False


async def test_search(session, query, search_type="hybrid"):
    """Test search functionality."""
    print(f"\n3. Testing {search_type} search: '{query}'")
    
    search_data = {
        "query": query,
        "top_k": 5,
        "search_type": search_type
    }
    
    async with session.post(f"{API_BASE_URL}/search", json=search_data) as resp:
        if resp.status == 200:
            result = await resp.json()
            print(f"✓ Search completed in {result['search_time_ms']:.2f}ms")
            print(f"  Found {result['total_results']} results:")
            
            for i, res in enumerate(result['results'], 1):
                print(f"\n  Result {i}:")
                print(f"    Score: {res['score']:.4f}")
                print(f"    Content: {res['content'][:150]}...")
                print(f"    Source: {res['metadata'].get('filename', 'Unknown')}")
                if res.get('highlights'):
                    print(f"    Highlights: {res['highlights'][0]}")
            
            return True
        else:
            error = await resp.text()
            print(f"✗ Search failed: {resp.status} - {error}")
            return False


async def test_index_refresh(session):
    """Test index refresh."""
    print("\n4. Testing index refresh...")
    
    async with session.post(f"{API_BASE_URL}/index/refresh") as resp:
        if resp.status == 200:
            result = await resp.json()
            print(f"✓ Index refresh completed:")
            print(f"  - Total files: {result['total_files']}")
            print(f"  - Indexed files: {result['indexed_files']}")
            print(f"  - Total chunks: {result['total_chunks']}")
            if result.get('errors'):
                print(f"  - Errors: {len(result['errors'])}")
                for error in result['errors'][:3]:
                    print(f"    • {error}")
            return True
        else:
            error = await resp.text()
            print(f"✗ Index refresh failed: {resp.status} - {error}")
            return False


async def test_stats(session):
    """Test collection statistics."""
    print("\n5. Testing collection stats...")
    
    async with session.get(f"{API_BASE_URL}/stats") as resp:
        if resp.status == 200:
            stats = await resp.json()
            print(f"✓ Collection statistics:")
            print(f"  - Collection: {stats['collection_name']}")
            print(f"  - Total documents: {stats['total_documents']}")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Index size: {stats['index_size_mb']} MB")
            return True
        else:
            print(f"✗ Stats request failed: {resp.status}")
            return False


async def test_ask_question(session, question):
    """Test the ask endpoint with natural language questions."""
    print(f"\n6. Testing question: '{question}'")
    
    payload = {
        "query": question,
        "top_k": 5,
        "include_sources": True,
        "temperature": 0.3
    }
    
    async with session.post(f"{API_BASE_URL}/ask", json=payload) as resp:
        if resp.status == 200:
            result = await resp.json()
            print(f"✓ Question answered successfully:")
            print(f"  - Query type: {result['query_type']}")
            print(f"  - Confidence: {result['confidence_score']:.2f}")
            print(f"  - Processing time: {result['processing_time_ms']:.2f}ms")
            print(f"    (Search: {result['search_time_ms']:.2f}ms, Generation: {result['generation_time_ms']:.2f}ms)")
            print(f"\n  Answer: {result['answer']}")
            
            if result.get('sources'):
                print(f"\n  Sources ({len(result['sources'])} found):")
                for i, source in enumerate(result['sources'][:3]):
                    print(f"    [{i+1}] Score: {source['relevance_score']:.3f}")
                    print(f"        {source['content'][:100]}...")
                    
            return True
        else:
            error = await resp.text()
            print(f"✗ Ask request failed: {resp.status} - {error}")
            return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Hybrid Search RAG API Test Suite")
    print("=" * 60)
    
    # Wait for services to be ready
    print("\nWaiting for services to start...")
    await asyncio.sleep(5)
    
    async with aiohttp.ClientSession() as session:
        # Test health check
        services = await test_health_check(session)
        if not services or not all(services.values()):
            print("\n⚠️  Not all services are healthy. Waiting 10 more seconds...")
            await asyncio.sleep(10)
            services = await test_health_check(session)
            if not services or not all(services.values()):
                print("\n❌ Services not ready. Please check Docker logs.")
                return
        
        # Upload sample files
        sample_files = list(EXAMPLE_DATA_DIR.glob("*"))[:3]  # Upload first 3 files
        if sample_files:
            print(f"\n📁 Found {len(sample_files)} sample files to upload")
            for file_path in sample_files:
                await test_upload_file(session, file_path)
                await asyncio.sleep(2)  # Give time for processing
        else:
            print("\n⚠️  No sample files found in example_data/")
        
        # Wait for indexing
        print("\n⏳ Waiting for indexing to complete...")
        await asyncio.sleep(5)
        
        # Test different search queries
        test_queries = [
            ("E-Commerce", "hybrid"),
            ("Workshop Manual Testing", "dense"),
            ("decidalo export", "sparse"),
            ("Trendradar", "hybrid")
        ]
        
        for query, search_type in test_queries:
            await test_search(session, query, search_type)
            await asyncio.sleep(1)
        
        # Test index refresh
        await test_index_refresh(session)
        
        # Get final stats
        await test_stats(session)
        
        # Test natural language questions
        test_questions = [
            "Who is Patrick Schönfeld?",
            "What is the role of decidalo in the company?",
            "Tell me about the workshop manual",
            "What products are mentioned in the documents?"
        ]
        
        for question in test_questions:
            await test_ask_question(session, question)
            await asyncio.sleep(1)
        
        print("\n" + "=" * 60)
        print("✅ Test suite completed!")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")