# Filename-based Deduplication Implementation

## Overview

The RAG system now implements filename-based deduplication to prevent duplicate entries when the same file is uploaded multiple times. When a file is uploaded, the system automatically removes any existing chunks associated with that filename before indexing the new version.

## How It Works

1. **Upload Endpoint**: When a file is uploaded via `/upload`:
   - The system first deletes all existing chunks with the same filename
   - Then processes and indexes the new file
   - Returns a message indicating if chunks were replaced

2. **Refresh Index Endpoint**: When refreshing the index via `/index/refresh`:
   - For each file in MinIO, existing chunks are deleted before re-indexing
   - This ensures no duplicates accumulate during refresh operations

## Implementation Details

### Vector Store Changes
- Added `delete_by_filename()` method to `VectorStore` class
- Uses Qdrant's filter functionality to find chunks by `metadata.source` field
- Performs batch deletion for efficiency

### Main API Changes
- Upload endpoint calls `delete_by_filename()` before indexing
- Refresh endpoint does the same for each file
- Response messages indicate when chunks are replaced

## Testing

Use the provided test script to verify deduplication:

```bash
python test_deduplication.py
```

The test will:
1. Upload a file and note the number of chunks created
2. Upload the same file again
3. Verify that the total chunk count only increased by the amount from a single upload
4. Confirm the file is still searchable

## Benefits

- **No Duplicates**: Each filename has only one version in the index
- **Automatic Updates**: Re-uploading a file automatically updates its content
- **Clean Index**: Prevents index bloat from multiple uploads
- **Consistent Search Results**: No duplicate results from the same source file