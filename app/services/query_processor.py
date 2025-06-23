import json
from typing import List, Dict, Optional
from loguru import logger
from models import QueryType


class QueryProcessor:
    """Process and optimize natural language queries using LLM for language-agnostic support."""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        
    async def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a natural language query using LLM for language-agnostic parsing.
        
        Returns:
            Dict containing:
            - query_type: Detected type of question
            - entities: Extracted entities
            - optimized_query: Optimized search query
            - expansion_terms: Additional search terms
            - language: Detected language
        """
        try:
            # If LLM service is available, use it for intelligent parsing
            if self.llm_service:
                return await self._process_with_llm(query)
            else:
                # Fallback to simple processing
                return self._simple_fallback(query)
                
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            # Return fallback result
            return self._simple_fallback(query)
    
    async def _process_with_llm(self, query: str) -> Dict[str, any]:
        """Process query using LLM for intelligent, multilingual parsing."""
        try:
            prompt = f"""Analyze this query in any language and extract the following information:

1. Query type: Classify as WHO (asking about a person), WHAT (asking for definition/description), WHERE (asking about location), WHEN (asking about time), WHY (asking for reasons), HOW (asking about methods), FACTUAL (asking for facts), or OTHER
2. Main entities: Extract key names, places, or things being asked about
3. Search query: Create an optimized search query by removing question words and keeping only the essential terms
4. Language: Detect the language code (en, de, es, fr, etc.)

Query: "{query}"

Return ONLY a JSON object in this exact format:
{{
  "query_type": "WHO",
  "entities": ["example entity"],
  "search_query": "optimized search terms",
  "language": "en",
  "expansion_terms": ["related", "terms"]
}}

Examples:
- "Who is Patrick Schönfeld?" → {{"query_type": "WHO", "entities": ["Patrick Schönfeld"], "search_query": "Patrick Schönfeld", "language": "en", "expansion_terms": ["profile", "employee", "role"]}}
- "Wer ist Patrick Schönfeld?" → {{"query_type": "WHO", "entities": ["Patrick Schönfeld"], "search_query": "Patrick Schönfeld", "language": "de", "expansion_terms": ["Profil", "Mitarbeiter", "Rolle"]}}
- "¿Dónde está la oficina de Munich?" → {{"query_type": "WHERE", "entities": ["oficina de Munich", "Munich"], "search_query": "Munich oficina office", "language": "es", "expansion_terms": ["ubicación", "dirección", "sede"]}}"""

            response = await self.llm_service.client.chat.completions.create(
                model=self.llm_service.deployment,
                messages=[
                    {"role": "system", "content": "You are a multilingual query parsing assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (sometimes LLM adds explanation)
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed_result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
            
            # Map string query type to enum
            query_type_str = parsed_result.get("query_type", "OTHER").upper()
            try:
                query_type = QueryType[query_type_str]
            except KeyError:
                query_type = QueryType.OTHER
            
            return {
                "query_type": query_type,
                "entities": parsed_result.get("entities", []),
                "optimized_query": parsed_result.get("search_query", query),
                "expansion_terms": parsed_result.get("expansion_terms", []),
                "language": parsed_result.get("language", "unknown"),
                "original_query": query
            }
            
        except Exception as e:
            logger.error(f"LLM query processing failed: {e}")
            raise
    
    def _simple_fallback(self, query: str) -> Dict[str, any]:
        """Simple fallback processing when LLM is not available."""
        # Remove common question marks and punctuation
        cleaned_query = query.strip().rstrip('?!.,;:')
        
        # Try to detect query type from first word
        first_word = query.lower().split()[0] if query.split() else ""
        query_type_map = {
            "who": QueryType.WHO, "wer": QueryType.WHO, "qui": QueryType.WHO, "quién": QueryType.WHO,
            "what": QueryType.WHAT, "was": QueryType.WHAT, "que": QueryType.WHAT, "qué": QueryType.WHAT,
            "where": QueryType.WHERE, "wo": QueryType.WHERE, "où": QueryType.WHERE, "dónde": QueryType.WHERE,
            "when": QueryType.WHEN, "wann": QueryType.WHEN, "quand": QueryType.WHEN, "cuándo": QueryType.WHEN,
            "why": QueryType.WHY, "warum": QueryType.WHY, "pourquoi": QueryType.WHY, "por": QueryType.WHY,
            "how": QueryType.HOW, "wie": QueryType.HOW, "comment": QueryType.HOW, "cómo": QueryType.HOW
        }
        
        query_type = query_type_map.get(first_word, QueryType.OTHER)
        
        # Extract potential entities (capitalized words)
        import re
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', query)
        
        # For optimized query, remove the question word if detected
        words = query.split()
        if words and words[0].lower() in query_type_map:
            optimized_query = ' '.join(words[1:])
        else:
            optimized_query = query
        
        return {
            "query_type": query_type,
            "entities": entities,
            "optimized_query": optimized_query.strip(),
            "expansion_terms": [],
            "language": "unknown",
            "original_query": query
        }