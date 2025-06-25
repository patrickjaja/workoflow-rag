#!/usr/bin/env python3
"""
Test script for MCP (Model Context Protocol) endpoint
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any


class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000", auth_token: str = None):
        self.base_url = base_url
        self.mcp_url = f"{base_url}/mcp"
        self.auth_token = auth_token
        self.request_id = 0
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    def _create_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        self.request_id += 1
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }
    
    async def call_method(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            request_data = self._create_request(method, params)
            print(f"\n🔹 Calling {method}...")
            print(f"Request: {json.dumps(request_data, indent=2)}")
            
            async with session.post(
                self.mcp_url,
                headers=self._get_headers(),
                json=request_data
            ) as response:
                result = await response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                return result


async def test_mcp_server():
    """Test all MCP server functionality"""
    client = MCPClient()
    
    print("=" * 80)
    print("🧪 Testing MCP Server Implementation")
    print("=" * 80)
    
    # Test 1: Initialize
    print("\n📍 Test 1: Initialize MCP Connection")
    result = await client.call_method("initialize")
    assert "result" in result
    assert "capabilities" in result["result"]
    print("✅ Initialize successful")
    
    # Test 2: List Tools
    print("\n📍 Test 2: List Available Tools")
    result = await client.call_method("tools/list")
    assert "result" in result
    assert "tools" in result["result"]
    tools = result["result"]["tools"]
    print(f"✅ Found {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
    
    # Test 3: Get Collection Stats
    print("\n📍 Test 3: Get Collection Statistics")
    result = await client.call_method("tools/call", {
        "name": "get_collection_stats",
        "arguments": {}
    })
    if "result" in result:
        print("✅ Collection stats retrieved successfully")
    else:
        print("⚠️  Collection might be empty")
    
    # Test 4: Search Documents
    print("\n📍 Test 4: Search Documents")
    result = await client.call_method("tools/call", {
        "name": "search_documents",
        "arguments": {
            "query": "Patrick Schönfeld",
            "top_k": 3,
            "search_type": "hybrid"
        }
    })
    if "result" in result:
        content = json.loads(result["result"]["content"][0]["text"])
        print(f"✅ Search returned {content['count']} results")
    else:
        print("⚠️  No search results found")
    
    # Test 5: Ask Question
    print("\n📍 Test 5: Ask a Question")
    result = await client.call_method("tools/call", {
        "name": "ask_question",
        "arguments": {
            "question": "Who is Patrick Schönfeld?",
            "top_k": 5,
            "include_sources": True
        }
    })
    if "result" in result:
        content = json.loads(result["result"]["content"][0]["text"])
        print(f"✅ Got answer: {content['answer'][:100]}...")
        print(f"   Confidence: {content['confidence']}")
    else:
        print("⚠️  Could not generate answer")
    
    # Test 6: List Resources
    print("\n📍 Test 6: List Resources")
    result = await client.call_method("resources/list")
    if "result" in result and "resources" in result["result"]:
        resources = result["result"]["resources"]
        print(f"✅ Found {len(resources)} resources")
        for res in resources[:3]:  # Show first 3
            print(f"   - {res['name']}")
    else:
        print("⚠️  No resources found")
    
    # Test 7: Invalid Method
    print("\n📍 Test 7: Test Error Handling (Invalid Method)")
    result = await client.call_method("invalid/method")
    if "error" in result:
        print(f"✅ Error handling works: {result['error']['message']}")
    else:
        print("❌ Error handling failed")
    
    print("\n" + "=" * 80)
    print("🎉 MCP Server Testing Complete!")
    print("=" * 80)


async def test_with_auth():
    """Test MCP server with authentication"""
    print("\n📍 Testing with Authentication")
    
    # This will fail if auth is enabled without proper token
    client = MCPClient(auth_token="test-token-123")
    result = await client.call_method("initialize")
    
    if "error" in result:
        print("✅ Authentication check works (request rejected)")
    else:
        print("✅ Authentication not required or token is valid")


if __name__ == "__main__":
    print("🚀 Starting MCP Server Tests")
    print("Make sure the RAG API is running on http://localhost:8000")
    print("-" * 80)
    
    try:
        # Run main tests
        asyncio.run(test_mcp_server())
        
        # Optionally test authentication
        # asyncio.run(test_with_auth())
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Make sure the RAG API is running and accessible")