import arxiv
import time
import logging
import os
import asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

async def fetch_papers(
    query: str,
    max_docs: int = 5,
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    retry_attempts: int = 3
) -> List[Dict[str, Any]]:
    """
    Fetch research papers from arXiv based on a query.
    
    Args:
        query: Search query for arXiv
        max_docs: Maximum number of documents to retrieve
        categories: Optional list of arXiv categories to filter by
        date_from: Optional date filter in format YYYY-MM-DD
        retry_attempts: Number of retry attempts if API fails
        
    Returns:
        List of dictionaries containing paper information
    """
    # Run in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(
        _fetch_papers_sync,
        query,
        max_docs,
        categories,
        date_from,
        retry_attempts
    )

def _fetch_papers_sync(
    query: str,
    max_docs: int = 5,
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    retry_attempts: int = 3
) -> List[Dict[str, Any]]:
    """Synchronous implementation of fetch_papers."""
    documents = []
    
    # Build more advanced query if filters are provided
    full_query = query
    if categories:
        category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
        full_query = f"({query}) AND ({category_filter})"
    if date_from:
        full_query = f"({full_query}) AND submittedDate:[{date_from}0000 TO 99990101000000]"
    
    logger.info(f"Searching arXiv for: '{full_query}'")
    
    attempt = 0
    while attempt < retry_attempts:
        try:
            client = arxiv.Client(
                page_size=100,  # Efficient batch size
                delay_seconds=3,  # Respect rate limits
                num_retries=3
            )
            
            search = arxiv.Search(
                query=full_query,
                max_results=max_docs,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = list(client.results(search))
            
            if not results:
                logger.warning(f"No papers found for query: '{query}'")
                return []
            
            for i, result in enumerate(results):
                # Create metadata with rich information
                paper = {
                    "id": f"arxiv_{result.entry_id.split('/')[-1]}",
                    "content": result.summary,
                    "metadata": {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "published": str(result.published),
                        "updated": str(result.updated) if result.updated else None,
                        "arxiv_id": result.entry_id.split("/")[-1],
                        "pdf_url": result.pdf_url,
                        "categories": result.categories,
                        "comment": result.comment,
                        "journal_ref": result.journal_ref
                    }
                }
                documents.append(paper)
            
            logger.info(f"Successfully fetched {len(documents)} papers")
            return documents
            
        except Exception as e:
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            logger.error(f"Error fetching papers (attempt {attempt}/{retry_attempts}): {str(e)}")
            if attempt < retry_attempts:
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retry attempts reached. Could not fetch papers.")
                return []
                
    return documents
