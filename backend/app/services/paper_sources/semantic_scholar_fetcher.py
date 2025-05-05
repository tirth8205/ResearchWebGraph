import logging
import os
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base_fetcher import PaperFetcher

load_dotenv()
logger = logging.getLogger(__name__)

class SemanticScholarFetcher(PaperFetcher):
    """Fetches papers from Semantic Scholar API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        if not self.api_key:
            logger.warning("No Semantic Scholar API key provided. Rate limits may apply.")
        
    @property
    def source_name(self) -> str:
        return "semantic_scholar"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying Semantic Scholar request (attempt {retry_state.attempt_number}/3)..."
        )
    )
    async def fetch_papers(
        self,
        query: str,
        max_docs: int = 5,
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from Semantic Scholar.
        
        Args:
            query: Search query
            max_docs: Maximum number of documents to retrieve
            categories: Optional list of categories (not used for Semantic Scholar)
            date_from: Optional date filter in format YYYY-MM-DD
            
        Returns:
            List of dictionaries containing paper information
        """
        papers = []
        
        # Validate query
        if not query or not query.strip():
            logger.error("Empty query provided to SemanticScholarFetcher")
            return []
        
        # Build query parameters
        params = {
            "query": query,
            "limit": max_docs,
            "fields": "title,authors,abstract,url,year,venue,publicationDate,fieldsOfStudy"
        }
        
        # Handle date filter
        current_year = datetime.now().year
        if date_from:
            try:
                year = int(date_from.split("-")[0])
                if year > current_year:
                    logger.warning(f"Future date_from {date_from}. Using range from 5 years ago.")
                    year = current_year - 5
                params["year"] = f"{year}-{current_year}"
            except ValueError:
                logger.warning(f"Invalid date_from format: {date_from}. Ignoring date filter.")
        else:
            # Default to 5 years ago to current year
            params["year"] = f"{current_year - 5}-{current_year}"
        
        # Headers with API key if provided
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            logger.info("Using Semantic Scholar API key for request")
        
        try:
            logger.info(f"Searching Semantic Scholar for: '{query}' with year filter: {params.get('year', 'none')}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/paper/search",
                    params=params,
                    headers=headers
                )
                
                response.raise_for_status()  # Raises for 4xx/5xx errors, including 429
                
                data = response.json()
                for paper in data.get("data", []):
                    # Transform to standard format
                    papers.append({
                        "id": f"semantic_{paper.get('paperId')}",
                        "content": paper.get("abstract", ""),
                        "metadata": {
                            "title": paper.get("title", "Untitled"),
                            "authors": [author.get("name", "") for author in paper.get("authors", [])],
                            "published": paper.get("publicationDate", str(paper.get("year", ""))),
                            "venue": paper.get("venue", ""),
                            "fields_of_study": paper.get("fieldsOfStudy", []),
                            "url": paper.get("url", ""),
                            "source": "semantic_scholar"
                        }
                    })
                
                logger.info(f"Successfully fetched {len(papers)} papers from Semantic Scholar")
                return papers
        except httpx.HTTPStatusError as e:
            error_msg = f"Semantic Scholar API error: {e.response.status_code}, {e.response.text}"
            logger.error(error_msg)
            raise  # Let tenacity handle retries
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {str(e)}")
            return []