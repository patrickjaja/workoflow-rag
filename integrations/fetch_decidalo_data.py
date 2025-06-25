#!/usr/bin/env python3
"""
Fetch Decidalo employee data and prepare it for RAG system upload.

This script fetches employee data from Decidalo API endpoints and creates
a structured JSON file optimized for the RAG system's search capabilities.
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
from functools import partial
import re
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecidaloDataFetcher:
    """Fetches and processes Decidalo employee data."""
    
    def __init__(self, bearer_token: str, cache_dir: str = ".cache"):
        self.bearer_token = bearer_token
        self.cache_dir = Path(__file__).parent / cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://api.decidalo.app/api"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
        
        # API endpoints for each employee
        self.employee_endpoints = {
            "projects": "/Profile/{user_id}/ProjectsSection?userID={user_id}",
            "trainings": "/Profile/{user_id}/TrainingsSection?userID={user_id}",
            "publications": "/Profile/{user_id}/PublicationsSection?userID={user_id}",
            "professional_experience": "/Profile/{user_id}/ProfessionalExperienceSection?userID={user_id}",
            "testimonials": "/Profile/{user_id}/TestimonialsSection?userID={user_id}",
            "employee_info": "/Profile/{user_id}/EmployeeInfoSection",
            "skills": "/Profile/{user_id}/Skills",
            "languages": "/Profile/{user_id}/LanguageSection",
            "roles": "/Profile/{user_id}/RolesSection",
            "industries": "/Profile/{user_id}/IndustrySection",
            "certificates": "/Profile/{user_id}/Certificates",
            "core_competencies": "/Profile/{user_id}/CoreCompetencies"
        }
        
    def _get_cache_path(self, endpoint: str, **kwargs) -> Path:
        """Generate cache file path for an endpoint."""
        # Replace placeholders in endpoint
        cache_key = endpoint
        for key, value in kwargs.items():
            cache_key = cache_key.replace(f"{{{key}}}", str(value))
        
        # Clean up the path for filesystem
        cache_key = cache_key.replace("/", "_").replace("?", "_")
        return self.cache_dir / f"{cache_key}.json"
    
    def _fetch_with_cache(self, url: str, cache_path: Path, method: str = "GET", 
                         json_data: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
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
                if method == "GET":
                    response = requests.get(url, headers=self.headers, timeout=30)
                else:
                    response = requests.post(url, headers=self.headers, json=json_data, timeout=30)
                
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
    
    def fetch_all_employees(self) -> Optional[List[Dict]]:
        """Fetch all employees from GetUsers endpoint."""
        logger.info("Fetching all employees...")
        
        url = f"{self.base_url}/User/GetUsers"
        payload = {
            "pageIndex": 0,
            "pageSize": 1100,  # Adjust if needed
            "pattern": "",
            "filters": [],
            "sort": {
                "viewMetamodelEntryID": 1,
                "sortDirection": 0
            }
        }
        
        cache_path = self._get_cache_path("User_GetUsers")
        data = self._fetch_with_cache(url, cache_path, method="POST", json_data=payload)
        
        if data and "data" in data:
            logger.info(f"Found {len(data['data'])} employees (total: {data.get('totalCount', 'unknown')})")
            return data["data"]
        
        return None
    
    def fetch_employee_details(self, user_id: int) -> Dict[str, Any]:
        """Fetch all details for a specific employee."""
        details = {"user_id": user_id}
        
        for endpoint_name, endpoint_template in self.employee_endpoints.items():
            endpoint = endpoint_template.format(user_id=user_id)
            url = f"{self.base_url}{endpoint}"
            cache_path = self._get_cache_path(endpoint, user_id=user_id)
            
            logger.debug(f"Fetching {endpoint_name} for user {user_id}")
            data = self._fetch_with_cache(url, cache_path)
            
            if data is not None:
                details[endpoint_name] = data
            else:
                details[endpoint_name] = None
        
        return details
    
    def extract_employee_info(self, employee_data: Dict) -> Dict[str, Any]:
        """Extract key information from employee data structure."""
        # The employee data structure uses numeric keys for fields
        info = {
            "id": employee_data.get("1", {}).get("id") if isinstance(employee_data.get("1"), dict) else None,
            "name": employee_data.get("1", {}).get("name") if isinstance(employee_data.get("1"), dict) else None,
            "email": employee_data.get("4"),
            "position": employee_data.get("209"),
            "company": employee_data.get("35", {}).get("name") if isinstance(employee_data.get("35"), dict) else None,
            "location": employee_data.get("36"),
            "phone": employee_data.get("38"),
            "career_level": employee_data.get("456"),  # Karrierestufe
            "summary": employee_data.get("224"),
            "summary_long": employee_data.get("482"),
            "linkedin_url": employee_data.get("371"),
            "highest_education": employee_data.get("370"),
            "nationality": employee_data.get("373"),
            "business_unit": employee_data.get("569"),
            "practice_area": employee_data.get("571"),
            "cost_center": employee_data.get("573"),
            "entry_date": employee_data.get("575"),
            "exit_date": employee_data.get("577"),
            "weekly_hours": employee_data.get("579"),
            "team": employee_data.get("415", {}).get("name") if isinstance(employee_data.get("415"), dict) else None,
            "manager": employee_data.get("417", {}).get("name") if isinstance(employee_data.get("417"), dict) else None,
            "country_code": employee_data.get("434"),
            "roles": employee_data.get("586"),
            "last_edit_date": employee_data.get("-4"),
            "profile_quality": employee_data.get("590")
        }
        
        # Remove None values
        return {k: v for k, v in info.items() if v is not None}
    
    def strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Remove HTML tags
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text).strip()
    
    def minimize_text(self, text: str, max_length: int = 200) -> str:
        """Minimize text to fit within character limit."""
        if not text:
            return ""
        
        # Strip HTML first
        clean_text = self.strip_html(text)
        
        # Truncate if needed
        if len(clean_text) > max_length:
            return clean_text[:max_length-3] + "..."
        
        return clean_text
    
    def build_employee_document(self, employee_data: Dict, details: Dict) -> Dict[str, Any]:
        """Build a comprehensive employee document for RAG indexing."""
        # Extract basic info
        basic_info = self.extract_employee_info(employee_data)
        
        # Build comprehensive document
        document = {
            "type": "employee_profile",
            "employee": basic_info,
            "details": {}
        }
        
        # Add all fetched details
        for key, value in details.items():
            if key != "user_id" and value is not None:
                document["details"][key] = value
        
        # Add searchable summary combining key information
        searchable_parts = []
        
        # Add name and position
        if basic_info.get("name"):
            searchable_parts.append(f"Name: {basic_info['name']}")
        if basic_info.get("position"):
            searchable_parts.append(f"Position: {basic_info['position']}")
        if basic_info.get("email"):
            searchable_parts.append(f"Email: {basic_info['email']}")
        if basic_info.get("team"):
            searchable_parts.append(f"Team: {basic_info['team']}")
        if basic_info.get("company"):
            searchable_parts.append(f"Company: {basic_info['company']}")
        if basic_info.get("location"):
            searchable_parts.append(f"Location: {basic_info['location']}")
        
        # Add skills if available
        if details.get("skills") and isinstance(details["skills"], list):
            skills = [skill.get("name", "") for skill in details["skills"] if isinstance(skill, dict)]
            if skills:
                searchable_parts.append(f"Skills: {', '.join(skills)}")
        
        # Add certifications if available  
        if details.get("certificates") and isinstance(details["certificates"], list):
            certs = [cert.get("name", "") for cert in details["certificates"] if isinstance(cert, dict)]
            if certs:
                searchable_parts.append(f"Certificates: {', '.join(certs)}")
        
        document["searchable_summary"] = " | ".join(searchable_parts)
        
        return document
    
    def build_minimized_employee_document(self, employee_data: Dict, details: Dict) -> Dict[str, Any]:
        """Build a focused employee document for Qdrant indexing with all relevant searchable data."""
        # Extract basic info
        basic_info = self.extract_employee_info(employee_data)
        
        # Build comprehensive but focused document
        doc = {
            "id": basic_info.get("id"),
            "name": basic_info.get("name", ""),
            "email": basic_info.get("email", ""),
            "position": basic_info.get("position", ""),
            "company": basic_info.get("company", ""),
            "location": basic_info.get("location", ""),
            "phone": basic_info.get("phone", ""),
            "team": basic_info.get("team", ""),
            "manager": basic_info.get("manager", ""),
            "career_level": basic_info.get("career_level", ""),
            "linkedin_url": basic_info.get("linkedin_url", ""),
            "nationality": basic_info.get("nationality", ""),
            "highest_education": basic_info.get("highest_education", ""),
        }
        
        # Add clean summary (remove HTML)
        if basic_info.get("summary"):
            doc["summary"] = self.strip_html(basic_info["summary"])
        
        # Add roles
        if details.get("roles") and isinstance(details["roles"], dict):
            roles_data = details["roles"].get("roles", [])
            if roles_data and isinstance(roles_data, list):
                doc["roles"] = [role.get("roleName", "") for role in roles_data if role.get("roleName")]
        
        # Add core competencies
        if details.get("core_competencies") and isinstance(details["core_competencies"], list):
            doc["core_competencies"] = [comp.get("displayText", "") for comp in details["core_competencies"] if comp.get("displayText")]
        
        # Add all skills with experience years
        if details.get("skills") and isinstance(details["skills"], dict):
            skills_data = details["skills"].get("skills", [])
            if skills_data and isinstance(skills_data, list):
                # Filter only skills with experience > 0
                experienced_skills = [
                    skill for skill in skills_data 
                    if skill.get("accumulatedExperienceInYears", 0) > 0
                ]
                
                # Sort by experience
                experienced_skills.sort(
                    key=lambda x: x.get("accumulatedExperienceInYears", 0), 
                    reverse=True
                )
                
                # Create two lists: top skills with experience, and all skill names
                top_skills_with_exp = []
                all_skill_names = []
                
                for skill in experienced_skills:
                    skill_name = skill.get("name", "")
                    if skill_name:
                        all_skill_names.append(skill_name)
                        # Include all skills with experience
                        exp_years = skill.get("accumulatedExperienceInYears", 0)
                        top_skills_with_exp.append({
                            "name": skill_name,
                            "years": round(exp_years, 1)
                        })
                
                if top_skills_with_exp:
                    doc["top_skills"] = top_skills_with_exp
                if all_skill_names:
                    doc["all_skills"] = all_skill_names
        
        # Add languages with proficiency levels
        if details.get("languages") and isinstance(details["languages"], dict):
            langs_data = details["languages"].get("languages", [])
            if langs_data and isinstance(langs_data, list):
                doc["languages"] = []
                for lang in langs_data:
                    if lang.get("name"):
                        lang_info = {"name": lang["name"]}
                        if lang.get("languageLevelID"):
                            lang_info["level"] = lang["languageLevelID"]
                        doc["languages"].append(lang_info)
        
        # Add industries
        if details.get("industries") and isinstance(details["industries"], dict):
            industries_data = details["industries"].get("industries", [])
            if industries_data and isinstance(industries_data, list):
                doc["industries"] = [ind.get("industryName", "") for ind in industries_data if ind.get("industryName")]
        
        # Add certificates with details
        if details.get("certificates") and isinstance(details["certificates"], list):
            doc["certificates"] = []
            for cert in details["certificates"]:
                if isinstance(cert, dict) and cert.get("certificateName"):
                    cert_info = {
                        "name": cert["certificateName"],
                        "issuer": cert.get("issuerOrganizationName", ""),
                        "year": cert.get("issueYear", "")
                    }
                    doc["certificates"].append(cert_info)
        
        # Add projects with more details
        if details.get("projects") and isinstance(details["projects"], dict):
            project_data = details["projects"].get("data", [])
            if project_data and isinstance(project_data, list):
                doc["project_count"] = len(project_data)
                doc["projects"] = []
                
                for proj in project_data:  # All projects
                    if isinstance(proj, dict):
                        project_info = {}
                        
                        # Get project name
                        if "191" in proj and isinstance(proj["191"], dict):
                            project_info["name"] = proj["191"].get("name", "")
                        
                        # Get role
                        if "27" in proj:
                            project_info["role"] = proj["27"]
                        
                        # Get dates
                        if "33" in proj:
                            project_info["start_date"] = proj["33"]
                        if "34" in proj:
                            project_info["end_date"] = proj["34"]
                        
                        # Get skills used
                        #if "Skills" in proj and isinstance(proj["Skills"], list):
                        #    project_info["skills"] = [s.get("name", "") for s in proj["Skills"] if s.get("name")]
                        
                        if project_info.get("name"):
                            doc["projects"].append(project_info)
        
        # Add professional experience
        if details.get("professional_experience") and isinstance(details["professional_experience"], dict):
            exp_data = details["professional_experience"].get("data", [])
            if exp_data and isinstance(exp_data, list):
                doc["experience_count"] = len(exp_data)
                doc["experience"] = []
                
                for exp in exp_data:  # All experiences
                    if isinstance(exp, dict):
                        exp_info = {}
                        if "73" in exp:
                            exp_info["company"] = exp["73"]
                        if "75" in exp:
                            exp_info["position"] = exp["75"]
                        if "485" in exp:
                            exp_info["start_date"] = exp["485"]
                        if "487" in exp:
                            exp_info["end_date"] = exp["487"]
                        
                        if exp_info.get("company") or exp_info.get("position"):
                            doc["experience"].append(exp_info)
        
        # Create a searchable text field combining all important information
        search_parts = [
            doc.get("name", ""),
            doc.get("email", ""),
            doc.get("position", ""),
            doc.get("company", ""),
            doc.get("location", ""),
            doc.get("team", ""),
        ]
        
        # Add skills to search text
        if "all_skills" in doc:
            search_parts.append(" ".join(doc["all_skills"]))  # All skills
        
        # Add roles
        if "roles" in doc:
            search_parts.extend(doc["roles"])
        
        # Add core competencies
        if "core_competencies" in doc:
            search_parts.extend(doc["core_competencies"])
        
        # Add industries
        if "industries" in doc:
            search_parts.extend(doc["industries"])
        
        # Add certificate names
        if "certificates" in doc:
            search_parts.extend([cert["name"] for cert in doc["certificates"]])
        
        # Add project names
        if "projects" in doc:
            search_parts.extend([proj["name"] for proj in doc["projects"] if proj.get("name")])
        
        doc["search_text"] = " | ".join(filter(None, search_parts))
        
        # Remove None values and empty strings/lists
        doc = {k: v for k, v in doc.items() if v and (not isinstance(v, list) or len(v) > 0)}
        
        return doc
    
    def process_all_employees(self, max_workers: int = 5, minimized: bool = False) -> List[Dict]:
        """Process all employees with parallel fetching."""
        employees = self.fetch_all_employees()
        if not employees:
            logger.error("Failed to fetch employees")
            return []
        
        logger.info(f"Processing {len(employees)} employees (minimized={minimized})...")
        
        documents = []
        
        # Process employees in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_employee = {}
            
            for employee in employees:
                # Extract user ID
                user_id = None
                if isinstance(employee.get("1"), dict):
                    user_id = employee["1"].get("id")
                
                if user_id:
                    future = executor.submit(self.fetch_employee_details, user_id)
                    future_to_employee[future] = employee
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_employee)):
                employee = future_to_employee[future]
                try:
                    details = future.result()
                    if minimized:
                        document = self.build_minimized_employee_document(employee, details)
                    else:
                        document = self.build_employee_document(employee, details)
                    documents.append(document)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(future_to_employee)} employees")
                        
                except Exception as e:
                    logger.error(f"Failed to process employee: {e}")
        
        logger.info(f"Successfully processed {len(documents)} employees")
        return documents
    
    def save_to_file(self, documents: List[Dict], output_file: str = "decidalo_employees.json", minimized: bool = False):
        """Save documents to JSON file."""
        output_data = documents
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(documents)} employee documents to {output_file}")
        
        # Print file size
        file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Output file size: {file_size:.2f} MB")
        
        # If minimized, show document size statistics
        if minimized:
            logger.info("\nDocument size statistics for focused employee data:")
            char_counts = []
            
            for doc in documents:
                json_str = json.dumps(doc, ensure_ascii=False, separators=(',', ':'))
                char_count = len(json_str)
                char_counts.append(char_count)
            
            avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0
            logger.info(f"Average character count: {avg_chars:.0f}")
            logger.info(f"Min character count: {min(char_counts) if char_counts else 0}")
            logger.info(f"Max character count: {max(char_counts) if char_counts else 0}")
            
            # Show average field counts
            field_counts = [len(doc.keys()) for doc in documents]
            avg_fields = sum(field_counts) / len(field_counts) if field_counts else 0
            logger.info(f"\nAverage fields per document: {avg_fields:.1f}")
            logger.info(f"Min fields: {min(field_counts) if field_counts else 0}")
            logger.info(f"Max fields: {max(field_counts) if field_counts else 0}")


def main():
    """Main function."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch Decidalo employee data for RAG system')
    parser.add_argument('--minimized', '-m', action='store_true', 
                        help='Create focused output optimized for Qdrant indexing with all searchable data')
    parser.add_argument('--output', '-o', type=str, 
                        help='Output filename (default: decidalo_employees.json or decidalo_employees_minimized.json)')
    parser.add_argument('--workers', '-w', type=int, default=5,
                        help='Number of parallel workers for fetching (default: 5)')
    args = parser.parse_args()
    
    # Get bearer token from environment
    bearer_token = os.environ.get("DECIDALO_BEARER_TOKEN")
    
    if not bearer_token:
        logger.error("No bearer token found. Set DECIDALO_BEARER_TOKEN environment variable.")
        return
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        filename = "decidalo_employees_minimized.json" if args.minimized else "decidalo_employees.json"
        output_file = str(Path(__file__).parent / filename)
    
    # Create fetcher instance
    fetcher = DecidaloDataFetcher(bearer_token)
    
    # Process all employees
    documents = fetcher.process_all_employees(max_workers=args.workers, minimized=args.minimized)
    
    if documents:
        # Save to file
        fetcher.save_to_file(documents, output_file, minimized=args.minimized)
        
        logger.info("\nData fetching complete!")
        logger.info(f"You can now upload '{output_file}' to your RAG system using:")
        logger.info(f"curl -X POST -F 'file=@{output_file}' http://localhost:8000/upload")
        
        if args.minimized:
            logger.info("\nNote: This is a focused output optimized for Qdrant indexing.")
            logger.info("Each employee record contains all relevant searchable data in a simplified structure.")
    else:
        logger.error("No documents were processed successfully")


if __name__ == "__main__":
    main()