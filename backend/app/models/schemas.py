from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class PaperRequest(BaseModel):
    """Request model for searching papers."""
    query: str = Field(..., description="Research topic to search for")
    max_papers: int = Field(5, description="Maximum number of papers to fetch")
    categories: Optional[List[str]] = Field(None, description="Categories to filter by")
    date_from: Optional[str] = Field(None, description="Date to filter from (YYYY-MM-DD)")

class PaperMetadata(BaseModel):
    """Metadata for a research paper."""
    title: str
    authors: List[str]
    published: str
    updated: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    categories: Optional[List[str]] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None

class Paper(BaseModel):
    """Model representing a research paper."""
    id: str
    content: str
    metadata: PaperMetadata

class PaperResponse(BaseModel):
    """Response model containing multiple papers."""
    papers: List[Paper]
    count: int

class GraphNode(BaseModel):
    """Model representing a node in the knowledge graph."""
    id: str
    label: str
    type: str
    metadata: Optional[Dict[str, Any]] = None

class GraphEdge(BaseModel):
    """Model representing an edge in the knowledge graph."""
    source: str
    target: str
    label: str
    weight: float = 1.0

class KnowledgeGraph(BaseModel):
    """Model representing the entire knowledge graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class QueryRequest(BaseModel):
    """Request model for querying the papers."""
    query: str
    papers_ids: List[str]

class QueryResponse(BaseModel):
    """Response model for a query answer."""
    answer: str
    sources: List[Dict[str, Any]]

class PDFUploadResponse(BaseModel):
    """Response model for PDF upload."""
    paper_id: str
    filename: str
    success: bool
    message: Optional[str] = None

class GraphVisualizationRequest(BaseModel):
    """Request model for graph visualization."""
    graph_id: str
    title: Optional[str] = "Research Knowledge Graph"
    height: Optional[str] = "800px"
    width: Optional[str] = "100%"
    
class GraphVisualizationResponse(BaseModel):
    """Response model for graph visualization."""
    html_content: str
    graph_id: str
