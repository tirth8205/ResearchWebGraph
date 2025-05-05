import arxiv
import time
import logging
import os
import asyncio
import random
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

from .base_fetcher import PaperFetcher

load_dotenv()
logger = logging.getLogger(__name__)

class ArXivFetcher(PaperFetcher):
    """Fetches papers from arXiv."""
    
    @property
    def source_name(self) -> str:
        return "arxiv"
    
    async def fetch_papers(
        self,
        query: str,
        max_docs: int = 5,
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        retry_attempts: int = 5
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
            self._fetch_papers_sync,
            query,
            max_docs,
            categories,
            date_from,
            retry_attempts
        )

    def _fetch_papers_sync(
        self,
        query: str,
        max_docs: int = 5,
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        retry_attempts: int = 5
    ) -> List[Dict[str, Any]]:
        """Synchronous implementation of fetch_papers with improved reliability."""
        documents = []
        
        # Validate inputs to avoid silent failures
        if not query or not query.strip():
            logger.error("Empty query provided to ArXivFetcher")
            return []
            
        # Build more advanced query if filters are provided
        full_query = self._construct_query(query, date_from, categories)
        
        # Log the complete query for debugging
        logger.info(f"Constructed full ArXiv query: '{full_query}' with max_docs={max_docs}")
        
        # Initialize with a small delay that will increase with each retry
        base_delay_seconds = 5.0  # Increased from 3.0 to handle API instability
        
        # Create a client with proper rate limiting
        client = arxiv.Client(
            page_size=100,  # Efficient batch size
            delay_seconds=base_delay_seconds,  # Respect rate limits
            num_retries=2   # Client-level retries for each request
        )
        
        # Track all exceptions for detailed error reporting
        exceptions = []
        
        # Outer retry loop with exponential backoff
        for attempt in range(retry_attempts):
            try:
                # Add jitter to avoid request clustering
                current_delay = base_delay_seconds * (2 ** attempt) + random.uniform(0, 1)
                
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt+1}/{retry_attempts} after {current_delay:.2f}s delay")
                    time.sleep(current_delay)
                
                # Try with a simpler query on later attempts
                if attempt > 2 and "AND" in full_query:
                    simple_query = query.strip()
                    logger.info(f"Trying simplified query on attempt {attempt+1}: '{simple_query}'")
                    search = arxiv.Search(
                        query=simple_query,
                        max_results=max_docs,
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )
                else:
                    # Create normal search
                    search = arxiv.Search(
                        query=full_query,
                        max_results=max_docs,
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )
                
                # Get results with timeout handling
                logger.debug(f"Executing search for attempt {attempt+1}")
                
                # Try to catch timeout issues
                start_time = time.time()
                results = list(client.results(search))
                end_time = time.time()
                
                # Log query time for performance monitoring
                logger.info(f"ArXiv query took {end_time - start_time:.2f} seconds")
                
                # Log results count
                logger.info(f"Attempt {attempt+1}: Received {len(results)} papers")
                
                # Check if results are empty - might be API flakiness
                if not results and attempt < retry_attempts - 1:
                    logger.warning(f"Empty results for query: '{full_query}', attempt {attempt+1}/{retry_attempts}. Will retry.")
                    
                    # Wait longer before retrying
                    time.sleep(current_delay * 1.5)
                    continue
                
                if not results:
                    logger.warning(f"No papers found for query: '{full_query}' after {attempt+1} attempts")
                    return []
                
                # Process results
                for i, result in enumerate(results):
                    # Extract and verify required fields
                    try:
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
                                "journal_ref": result.journal_ref,
                                "source": "arxiv"
                            }
                        }
                        documents.append(paper)
                    except Exception as e:
                        logger.error(f"Error processing paper {i+1}: {str(e)}")
                        # Continue processing other papers
                        continue
                
                logger.info(f"Successfully fetched {len(documents)} papers from arXiv")
                return documents
                
            except Exception as e:
                # Detailed error reporting
                error_type = type(e).__name__
                error_msg = str(e)
                exceptions.append(f"{error_type}: {error_msg}")
                
                logger.error(f"Error fetching papers from arXiv (attempt {attempt+1}/{retry_attempts}): {error_type}, {error_msg}")
                
                # On last attempt, return empty list with detailed error
                if attempt == retry_attempts - 1:
                    all_errors = "; ".join(exceptions)
                    logger.error(f"Max retry attempts reached. Could not fetch papers from arXiv. Errors: {all_errors}")
                    return []
                
                # Wait before retrying with exponential backoff
                retry_delay = base_delay_seconds * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {retry_delay:.2f} seconds...")
                time.sleep(retry_delay)
        
        return documents

    def _construct_query(self, query: str, date_from: Optional[str] = None, categories: Optional[List[str]] = None) -> str:
        """
        Construct the full ArXiv query string with optional date and category filters.
        
        Args:
            query: The search query string
            date_from: Optional start date in YYYY-MM-DD format
            categories: Optional list of arXiv categories
            
        Returns:
            Formatted ArXiv query string
        """
        # Clean the query
        query = query.strip().replace(" ", "+")
        
        # Initialize query components
        query_components = [f"({query})"]
        
        # Handle date filter
        current_date = datetime.now()
        end_date = current_date.strftime("%Y%m%d%H%M")
        
        if date_from:
            try:
                # Parse provided date_from
                date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                if date_from_dt > current_date:
                    # If date_from is in the future, use a 5-year range ending today
                    date_from = (current_date - timedelta(days=5*365)).strftime("%Y%m%d%H%M")
                else:
                    date_from = date_from_dt.strftime("%Y%m%d%H%M")
            except ValueError:
                logger.warning(f"Invalid date_from format: {date_from}. Using 5-year range.")
                date_from = (current_date - timedelta(days=5*365)).strftime("%Y%m%d%H%M")
        else:
            # Default to 5 years ago
            date_from = (current_date - timedelta(days=5*365)).strftime("%Y%m%d%H%M")
        
        query_components.append(f"submittedDate:[{date_from} TO {end_date}]")
        
        # Handle categories
        if categories:
            formatted_categories = [cat.strip().replace(".", "_") for cat in categories if cat.strip()]
            if formatted_categories:
                category_query = " OR ".join(f"cat:{cat}" for cat in formatted_categories)
                query_components.append(f"({category_query})")
        
        # Combine components
        full_query = " AND ".join(query_components)
        
        logger.info(f"Constructed full ArXiv query: '{full_query}' with max_docs={self.max_docs}")
        return full_query