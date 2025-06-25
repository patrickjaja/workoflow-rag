# Custom Integrations

This directory contains custom integration scripts for fetching and preparing external data for the RAG system.

## Service Map Integration

The Service Map integration fetches services and case studies data from the Service Map API.

### Script: `integrations/fetch_service_map.py`

**Purpose**: Fetches services and case studies data from Service Map API endpoints and creates a structured JSON file optimized for RAG search.

**Usage**:
```bash
# Basic usage (requires SERVICE_MAP_BEARER_TOKEN in .env)
python integrations/fetch_service_map.py

# Custom output file
python integrations/fetch_service_map.py --output custom_output.json

# Adjust page size for API requests
python integrations/fetch_service_map.py --page-size 100
```

**Environment Variables**:
- `SERVICE_MAP_BEARER_TOKEN`: Bearer token for Service Map API authentication

**Output**: 
- Default: `integrations/service_map_data.json`
- Contains services and case studies with metadata for search

**Features**:
- Caches API responses in `integrations/.cache/` to avoid redundant requests
- Parallel fetching of services and case studies
- Transforms data into a simplified structure optimized for search

## Decidalo Employee Data Integration

The Decidalo integration fetches comprehensive employee profile data from the Decidalo API.

### Script: `integrations/fetch_decidalo_data.py`

**Purpose**: Fetches employee profiles with skills, projects, certifications, and other professional data from Decidalo API.

**Usage**:
```bash
# Full employee data (requires DECIDALO_BEARER_TOKEN in .env)
python integrations/fetch_decidalo_data.py

# Minimized/focused output optimized for Qdrant
python integrations/fetch_decidalo_data.py --minimized

# Custom output file
python integrations/fetch_decidalo_data.py --output custom_employees.json

# Adjust parallel workers
python integrations/fetch_decidalo_data.py --workers 10
```

**Environment Variables**:
- `DECIDALO_BEARER_TOKEN`: Bearer token for Decidalo API authentication

**Output**:
- Default: `integrations/decidalo_employees.json` (full data)
- Minimized: `integrations/decidalo_employees_minimized.json` (focused data)

**Features**:
- Fetches comprehensive employee profiles including:
  - Basic info (name, email, position, team, manager)
  - Skills with experience years
  - Projects with roles and dates
  - Professional experience
  - Certifications
  - Languages
  - Core competencies
- Parallel fetching with configurable workers
- Two output modes:
  - Full: Complete employee data for comprehensive indexing
  - Minimized: Focused data optimized for search with reduced size
- Caches API responses in `integrations/.cache/`

## Cache Management

Both scripts use a shared cache directory at `integrations/.cache/` to store API responses and avoid redundant requests. The cache:
- Significantly speeds up subsequent runs
- Reduces API load
- Preserves data during development

To clear the cache:
```bash
rm -rf integrations/.cache/
```

## Uploading to RAG System

After generating the data files, upload them to the RAG system:

```bash
# Upload Service Map data
curl -X POST -F 'file=@integrations/service_map_data.json' http://localhost:8000/upload

# Upload Decidalo employee data
curl -X POST -F 'file=@integrations/decidalo_employees_minimized.json' http://localhost:8000/upload
```

## Adding New Integrations

When adding new integration scripts:
1. Place them in the `integrations/` directory
2. Use the same cache directory pattern: `Path(__file__).parent / ".cache"`
3. Generate output files in the `integrations/` directory
4. Add the output files to `.gitignore`
5. Document the integration in this file