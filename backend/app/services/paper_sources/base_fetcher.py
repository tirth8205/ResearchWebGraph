from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class PaperFetcher(ABC):
    """Abstract base class for paper fetchers from different sources."""
    
    @abstractmethod
    async def fetch_papers(
        self,
        query: str,
        max_docs: int = 5,
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers based on search criteria.
        
        Args:
            query: Search query
            max_docs: Maximum number of documents to retrieve
            categories: Optional list of categories to filter by
            date_from: Optional date filter in format YYYY-MM-DD
            
        Returns:
            List of paper dictionaries with standardized format
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this paper source."""
        pass
