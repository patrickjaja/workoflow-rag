#!/usr/bin/env python3
"""
Fetch Service Map data and prepare it for RAG system upload.

This script fetches services and case studies data from Service Map API endpoints
and creates a structured JSON file optimized for the RAG system's search capabilities.
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceMapDataFetcher:
    """Fetches and processes Service Map data."""
    
    def __init__(self, bearer_token: str, cache_dir: str = ".cache"):
        self.bearer_token = bearer_token
        self.cache_dir = Path(__file__).parent / cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://backend.cx-service-map.vcec.cloud/api"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
        
    def _get_cache_path(self, endpoint: str, **kwargs) -> Path:
        """Generate cache file path for an endpoint."""
        # Build cache key from endpoint and parameters
        cache_key = endpoint.replace("/", "_")
        
        # Add parameters to cache key
        for key, value in sorted(kwargs.items()):
            cache_key += f"_{key}={value}"
        
        # Clean up the path for filesystem
        cache_key = cache_key.replace("[", "_").replace("]", "_").replace("*", "all")
        return self.cache_dir / f"{cache_key}.json"
    
    def _fetch_with_cache(self, url: str, cache_path: Path, max_retries: int = 3) -> Optional[Dict]:
        """Fetch data with caching and retry logic."""
        # Check cache first
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        
        # Fetch from API
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Save to cache
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save cache {cache_path}: {e}")
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
    
    def fetch_all_services(self, page_size: int = 90) -> List[Dict]:
        """Fetch all services with pagination."""
        logger.info("Fetching all services...")
        all_services = []
        page = 1
        
        while True:
            url = f"{self.base_url}/services?pagination[page]={page}&pagination[pageSize]={page_size}"
            cache_path = self._get_cache_path("_api_services", page=page, pageSize=page_size)
            
            logger.debug(f"Fetching services page {page}")
            data = self._fetch_with_cache(url, cache_path)
            
            if not data or "data" not in data or not data["data"]:
                break
            
            all_services.extend(data["data"])
            
            # Check if we have more pages
            meta = data.get("meta", {}).get("pagination", {})
            total_pages = meta.get("pageCount", 1)
            
            if page >= total_pages:
                break
            
            page += 1
        
        logger.info(f"Fetched {len(all_services)} services")
        return all_services
    
    def fetch_all_case_studies(self, page_size: int = 90) -> List[Dict]:
        """Fetch all case studies with pagination."""
        logger.info("Fetching all case studies...")
        all_case_studies = []
        page = 1
        
        while True:
            url = f"{self.base_url}/case-studies?pagination[page]={page}&pagination[pageSize]={page_size}&populate=*"
            cache_path = self._get_cache_path("_api_case-studies", page=page, pageSize=page_size, populate="all")
            
            logger.debug(f"Fetching case studies page {page}")
            data = self._fetch_with_cache(url, cache_path)
            
            if not data or "data" not in data or not data["data"]:
                break
            
            all_case_studies.extend(data["data"])
            
            # Check if we have more pages
            meta = data.get("meta", {}).get("pagination", {})
            total_pages = meta.get("pageCount", 1)
            
            if page >= total_pages:
                break
            
            page += 1
        
        logger.info(f"Fetched {len(all_case_studies)} case studies")
        return all_case_studies
    
    def transform_service(self, service: Dict) -> Dict[str, Any]:
        """Transform service data to minimized format."""
        attrs = service.get("attributes", {})
        
        return {
            "type": "service",
            "id": f"service_{service.get('id')}",
            "service_id": service.get("id"),
            "name": attrs.get("name", ""),
            "description": attrs.get("description", ""),
            "slug": attrs.get("slug", ""),
            "position": attrs.get("position", ""),
            "deliverables": attrs.get("deliverables", ""),
            "createdAt": attrs.get("createdAt", "")
        }
    
    def transform_case_study(self, case_study: Dict) -> Dict[str, Any]:
        """Transform case study data to minimized format."""
        attrs = case_study.get("attributes", {})
        
        # Extract services
        services = []
        services_data = attrs.get("services", {}).get("data", [])
        for service in services_data:
            service_attrs = service.get("attributes", {})
            services.append({
                "id": service.get("id"),
                "name": service_attrs.get("name", ""),
                "description": service_attrs.get("description", ""),
                "slug": service_attrs.get("slug", ""),
                "position": service_attrs.get("position", ""),
                "deliverables": service_attrs.get("deliverables", "")
            })
        
        # Extract contact
        contact = None
        contact_data = attrs.get("contact", {}).get("data")
        if contact_data:
            contact_attrs = contact_data.get("attributes", {})
            contact = {
                "id": contact_data.get("id"),
                "name": contact_attrs.get("name", ""),
                "position": contact_attrs.get("position", ""),
                "biography": contact_attrs.get("biography", ""),
                "mail": contact_attrs.get("mail", "")
            }
        
        # Extract technologies
        technologies = []
        tech_data = attrs.get("technologies", {}).get("data", [])
        for tech in tech_data:
            tech_attrs = tech.get("attributes", {})
            technologies.append({
                "id": tech.get("id"),
                "name": tech_attrs.get("name", ""),
                "url": tech_attrs.get("url", "")
            })
        
        # Extract customer industry
        customer_industry = None
        industry_data = attrs.get("customer_industry", {}).get("data")
        if industry_data:
            customer_industry = industry_data.get("attributes", {}).get("name", "")
        
        return {
            "type": "case_study",
            "id": f"case_study_{case_study.get('id')}",
            "case_study_id": case_study.get("id"),
            "title": attrs.get("title", ""),
            "key_results": attrs.get("key_results", ""),
            "pain": attrs.get("pain", ""),
            "solution": attrs.get("solution", ""),
            "client_description": attrs.get("client_description", ""),
            "slug": attrs.get("slug", ""),
            "buying_center": attrs.get("buying_center", ""),
            "services": services,
            "contact": contact,
            "technologies": technologies,
            "customer_industry": customer_industry
        }
    
    def process_all_data(self) -> List[Dict]:
        """Process all services and case studies."""
        logger.info("Processing all Service Map data...")
        
        # Fetch services and case studies in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            services_future = executor.submit(self.fetch_all_services)
            case_studies_future = executor.submit(self.fetch_all_case_studies)
            
            services = services_future.result()
            case_studies = case_studies_future.result()
        
        # Transform data
        documents = []
        
        # Transform services
        for service in services:
            try:
                doc = self.transform_service(service)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to transform service {service.get('id')}: {e}")
        
        # Transform case studies
        for case_study in case_studies:
            try:
                doc = self.transform_case_study(case_study)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to transform case study {case_study.get('id')}: {e}")
        
        logger.info(f"Successfully processed {len(documents)} documents")
        return documents
    
    def save_to_file(self, documents: List[Dict], output_file: str = "service_map_data.json"):
        """Save documents to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(documents)} documents to {output_file}")
        
        # Print file size
        file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Output file size: {file_size:.2f} MB")
        
        # Show document type statistics
        service_count = sum(1 for doc in documents if doc["type"] == "service")
        case_study_count = sum(1 for doc in documents if doc["type"] == "case_study")
        
        logger.info(f"\nDocument breakdown:")
        logger.info(f"- Services: {service_count}")
        logger.info(f"- Case Studies: {case_study_count}")


def main():
    """Main function."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch Service Map data for RAG system')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (default: integrations/service_map_data.json)')
    parser.add_argument('--page-size', '-p', type=int, default=90,
                        help='Page size for API requests (default: 90)')
    args = parser.parse_args()
    
    # Get bearer token from environment
    bearer_token = os.environ.get("SERVICE_MAP_BEARER_TOKEN")
    
    if not bearer_token:
        logger.error("No bearer token found. Set SERVICE_MAP_BEARER_TOKEN environment variable.")
        return
    
    # Create fetcher instance
    fetcher = ServiceMapDataFetcher(bearer_token)
    
    # Process all data
    documents = fetcher.process_all_data()
    
    if documents:
        # Determine output file
        output_file = args.output if args.output else str(Path(__file__).parent / "service_map_data.json")
        
        # Save to file
        fetcher.save_to_file(documents, output_file)
        
        logger.info("\nData fetching complete!")
        logger.info(f"You can now upload '{output_file}' to your RAG system using:")
        logger.info(f"curl -X POST -F 'file=@{output_file}' http://localhost:8000/upload")
    else:
        logger.error("No documents were processed successfully")


if __name__ == "__main__":
    main()