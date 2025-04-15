import logging
import os
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .base_fetcher import PaperFetcher

load_dotenv()
logger = logging.getLogger(__name__)

class SemanticScholarFetcher(PaperFetcher):
    """Fetches papers from Semantic Scholar API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
    @property
    def source_name(self) -> str:
        return "semantic_scholar"
    
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
        
        # Build query parameters
        params = {
            "query": query,
            "limit": max_docs,
            "fields": "title,authors,abstract,url,year,venue,publicationDate,fieldsOfStudy"
        }
        
        # Add filters for date if provided
        if date_from:
            year = date_from.split("-")[0]
            params["year"] = f"{year}-"
        
        # Headers with API key if provided
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            logger.info(f"Searching Semantic Scholar for: '{query}'")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/paper/search",
                    params=params,
                    headers=headers
                )
                
                if response.status_code == 200:
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
                else:
                    error_msg = f"Semantic Scholar API error: {response.status_code}, {response.text}"
                    logger.error(error_msg)
                
                return papers
        except Exception as e:
            logger.error(f"Error fetching from Semantic Scholar: {str(e)}")
            return []
