# Remote Debugging with PyCharm

This guide shows how to debug the containerized FastAPI application using PyCharm's remote debugging feature.

## Setup

### 1. Rebuild the Docker image
```bash
docker-compose build app
```

### 2. Configure PyCharm Remote Debug

1. In PyCharm, go to **Run → Edit Configurations**
2. Click **+** and select **Python Remote Debug**
3. Configure as follows:
   - **Name**: Docker RAG Debug
   - **Host**: localhost
   - **Port**: 5678
   - **Path mappings**: 
     - Local path: `/home/patrickjaja/development/python-rag/app`
     - Remote path: `/app`

### 3. Start Debugging

#### Option 1: Always wait for debugger (recommended for debugging)
```bash
# Use the debug compose file
docker-compose -f docker-compose.debug.yml up
```
The container will start and wait for the debugger to attach before starting the API.

#### Option 2: Conditional debugging
```bash
# Add debug port to regular docker-compose.yml
# Add to app service:
#   ports:
#     - "5678:5678"
#   environment:
#     - DEBUG_MODE=true

# Then use the debug startup script
docker-compose run --rm app python debug_main.py
```

### 4. Debug Workflow

1. Set your breakpoint in PyCharm (e.g., at `app/services/answer_generator.py:110`)
2. Start the PyCharm remote debug server (click the debug button for your configuration)
3. Start the Docker container using one of the methods above
4. Wait for "Debugger attached!" message in the container logs
5. Make your API call: `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "test"}'`
6. PyCharm will stop at your breakpoint

## Troubleshooting

### Container won't start
- Ensure port 5678 is not already in use: `lsof -i :5678`
- Check Docker logs: `docker-compose logs -f app`

### Debugger won't connect
- Verify PyCharm remote debug server is running
- Check firewall settings for port 5678
- Ensure path mappings are correct in PyCharm configuration

### Breakpoints not hit
- Verify the file path mapping is correct
- Ensure you're hitting the correct endpoint that triggers the code
- Check that the breakpoint is on an executable line

## Alternative: VS Code Debugging

If using VS Code, create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/app",
                    "remoteRoot": "/app"
                }
            ]
        }
    ]
}
```