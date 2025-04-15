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
        self.sources[fetcher.source_name] = fetcher
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
        if not sources:
            # If no sources specified, use all available sources
            sources = list(self.sources.keys())
        
        # Limit to available sources
        sources = [s for s in sources if s in self.sources]
        
        if not sources:
            logger.warning("No valid paper sources specified")
            return []
        
        # Create fetch tasks for each source
        tasks = []
        for source_name in sources:
            fetcher = self.sources[source_name]
            task = fetcher.fetch_papers(
                query=query,
                max_docs=max_docs_per_source,
                categories=categories,
                date_from=date_from
            )
            tasks.append(task)
        
        # Run all fetch tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results, handling any exceptions
        all_papers = []
        for source_name, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching from {source_name}: {str(result)}")
            else:
                all_papers.extend(result)
                logger.info(f"Fetched {len(result)} papers from {source_name}")
        
        return all_papers
