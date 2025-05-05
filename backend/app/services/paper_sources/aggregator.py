from typing import List, Dict, Any, Optional
import asyncio
import logging
import os
import time

from .base_fetcher import PaperFetcher
from .arxiv_fetcher import ArXivFetcher
from .semantic_scholar_fetcher import SemanticScholarFetcher

logger = logging.getLogger(__name__)

class PaperSourceAggregator:
    """Aggregates results from multiple paper sources."""
    
    def __init__(self):
        self.sources = {}
        
        # Register available sources
        self.register_source(ArXivFetcher())
        self.register_source(SemanticScholarFetcher())
    
    def register_source(self, fetcher: PaperFetcher):
        """Register a paper source."""
        self.sources[fetcher.source_name.lower()] = fetcher
        logger.info(f"Registered paper source: {fetcher.source_name}")
    
    async def fetch_papers(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_docs_per_source: int = 5,
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from multiple sources.
        
        Args:
            query: Search query
            sources: List of source names to search (None for all sources)
            max_docs_per_source: Maximum papers to fetch from each source
            categories: Optional categories to filter by
            date_from: Optional date to filter from
            
        Returns:
            Combined list of papers from all sources
        """
        if not query or query.strip() == "":
            logger.warning("Empty query provided to fetch_papers")
            return []
            
        # If no sources specified, use all available sources
        if not sources:
            sources = list(self.sources.keys())
        
        # Make source names case-insensitive for better matching
        sources = [s.lower() for s in sources if s]
        
        # Limit to available sources
        valid_sources = [s for s in sources if s in self.sources]
        
        if not valid_sources:
            logger.warning(f"No valid paper sources specified. Available sources: {list(self.sources.keys())}")
            # Default to all sources if none specified are valid
            valid_sources = list(self.sources.keys())
        
        # Create fetch tasks for each source
        tasks = []
        task_sources = []
        for source_name in valid_sources:
            fetcher = self.sources[source_name]
            
            # Handle parameter name differences between fetchers
            if source_name.lower() == "arxiv":
                task = fetcher.fetch_papers(
                    query=query,
                    max_results=max_docs_per_source,  # ArXiv uses max_results
                    categories=categories,
                    date_from=date_from
                )
            else:
                task = fetcher.fetch_papers(
                    query=query,
                    max_docs=max_docs_per_source,  # Others use max_docs
                    categories=categories,
                    date_from=date_from
                )
                
            tasks.append(task)
            task_sources.append(source_name)
        
        # Run all fetch tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results, handling any exceptions
        all_papers = []
        successful_sources = 0
        
        for source_name, result in zip(task_sources, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching from {source_name}: {str(result)}")
            else:
                all_papers.extend(result)
                successful_sources += 1
                logger.info(f"Fetched {len(result)} papers from {source_name}")
        
        # Log summary
        if not all_papers:
            if successful_sources == 0:
                logger.warning("All paper sources failed to return results")
            else:
                logger.warning("No papers found matching the query criteria")
        
        return all_papers
    
    def get_available_sources(self) -> List[str]:
        """
        Get a list of available paper sources.
        
        Returns:
            List of source names
        """
        return list(self.sources.keys())
