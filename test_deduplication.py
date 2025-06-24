#!/usr/bin/env python3
"""
Test script to verify filename-based deduplication in the RAG system.
"""

import asyncio
import httpx
import json
import time
from pathlib import Path


async def test_deduplication():
    """Test that uploading the same file twice replaces the old version."""
    
    base_url = "http://localhost:8000"
    client = httpx.AsyncClient(timeout=30.0)
    
    try:
        # Check health
        print("1. Checking system health...")
        health_response = await client.get(f"{base_url}/health")
        print(f"Health status: {health_response.json()}")
        print()
        
        # Get initial stats
        print("2. Getting initial collection stats...")
        initial_stats = await client.get(f"{base_url}/stats")
        initial_data = initial_stats.json()
        print(f"Initial chunks in collection: {initial_data['total_chunks']}")
        print()
        
        # Find a test file
        test_files = list(Path("example_data").glob("*.pdf"))[:1]
        if not test_files:
            print("No PDF files found in example_data directory")
            return
        
        test_file = test_files[0]
        print(f"3. Using test file: {test_file.name}")
        
        # First upload
        print("\n4. First upload of the file...")
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/pdf')}
            upload1 = await client.post(f"{base_url}/upload", files=files)
        
        upload1_data = upload1.json()
        print(f"First upload result: {upload1_data['message']}")
        print(f"Chunks created: {upload1_data['chunks_created']}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Check stats after first upload
        print("\n5. Checking stats after first upload...")
        stats1 = await client.get(f"{base_url}/stats")
        stats1_data = stats1.json()
        print(f"Total chunks after first upload: {stats1_data['total_chunks']}")
        
        # Second upload of the same file
        print(f"\n6. Second upload of the same file ({test_file.name})...")
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/pdf')}
            upload2 = await client.post(f"{base_url}/upload", files=files)
        
        upload2_data = upload2.json()
        print(f"Second upload result: {upload2_data['message']}")
        print(f"Chunks created: {upload2_data['chunks_created']}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Check final stats
        print("\n7. Checking final stats...")
        final_stats = await client.get(f"{base_url}/stats")
        final_data = final_stats.json()
        print(f"Total chunks after second upload: {final_data['total_chunks']}")
        
        # Verify deduplication worked
        print("\n8. Verification:")
        chunks_increase = final_data['total_chunks'] - initial_data['total_chunks']
        print(f"Net increase in chunks: {chunks_increase}")
        
        if chunks_increase == upload1_data['chunks_created']:
            print("✅ SUCCESS: Deduplication working correctly!")
            print("   The second upload replaced the first one.")
        else:
            print("❌ FAILURE: Deduplication not working!")
            print(f"   Expected net increase: {upload1_data['chunks_created']}")
            print(f"   Actual net increase: {chunks_increase}")
        
        # Test search to ensure the file is still searchable
        print("\n9. Testing search functionality...")
        search_query = test_file.stem.split('_')[0]  # Use first part of filename
        search_response = await client.post(
            f"{base_url}/search",
            json={"query": search_query, "top_k": 5}
        )
        search_data = search_response.json()
        print(f"Search for '{search_query}' returned {search_data['total_results']} results")
        
        if search_data['total_results'] > 0:
            print("✅ File is still searchable after replacement")
        else:
            print("❌ File not found in search after replacement")
        
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    print("Testing Filename-based Deduplication in RAG System")
    print("=" * 50)
    asyncio.run(test_deduplication())