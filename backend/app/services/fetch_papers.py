import time
import logging
import os
import asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Import the new aggregator
from app.services.paper_sources.aggregator import PaperSourceAggregator

load_dotenv()
logger = logging.getLogger(__name__)

async def fetch_papers(
    query: str,
    max_docs: int = 5,
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    sources: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch research papers from multiple sources based on a query.
    
    Args:
        query: Search query
        max_docs: Maximum number of documents to retrieve per source
        categories: Optional list of categories to filter by
        date_from: Optional date filter in format YYYY-MM-DD
        sources: Optional list of sources to use (defaults to all)
        
    Returns:
        List of dictionaries containing paper information
    """
    try:
        # Use the aggregator to fetch from multiple sources
        aggregator = PaperSourceAggregator()
        
        papers = await aggregator.fetch_papers(
            query=query,
            sources=sources,
            max_docs_per_source=max_docs,
            categories=categories,
            date_from=date_from
        )
        
        logger.info(f"Total papers fetched from all sources: {len(papers)}")
        return papers
        
    except Exception as e:
        logger.error(f"Error in fetch_papers: {str(e)}", exc_info=True)
        return []
