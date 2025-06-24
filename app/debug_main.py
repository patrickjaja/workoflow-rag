"""
Debug startup script for conditional debugging support.
Use this instead of main.py when you want debugging to be optional.
"""
import os
import sys

# Check if debug mode is enabled
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    try:
        import debugpy
        
        # Configure debugpy
        debug_port = int(os.getenv("DEBUG_PORT", "5678"))
        debugpy.listen(("0.0.0.0", debug_port))
        
        print(f"🔍 Debug mode enabled. Listening on port {debug_port}...")
        
        # Optional: wait for debugger to attach
        if os.getenv("DEBUG_WAIT_FOR_CLIENT", "true").lower() == "true":
            print("⏳ Waiting for debugger to attach...")
            debugpy.wait_for_client()
            print("✅ Debugger attached!")
    except ImportError:
        print("⚠️  debugpy not installed. Running without debug support.")
else:
    print("ℹ️  Debug mode disabled. Set DEBUG_MODE=true to enable.")

# Import and run the main app
from main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )