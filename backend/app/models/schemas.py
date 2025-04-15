from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID

class PaperRequest(BaseModel):
    query: str
    max_papers: int = 5
    categories: Optional[List[str]] = None
    date_from: Optional[str] = None
    sources: Optional[List[str]] = None  # New field for specifying sources

class PaperResponse(BaseModel):
    papers: List[Dict[str, Any]]
    count: int
    task_id: Optional[str] = None

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    metadata: Optional[Dict[str, Any]] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    weight: float = 1.0

class KnowledgeGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    id: Optional[str] = None
    created_at: Optional[str] = None

class GraphRequest(BaseModel):
    paper_ids: List[str]

class GraphResponse(BaseModel):
    graph: KnowledgeGraph
    paper_count: int
    node_count: int
    edge_count: int

class VisualizationRequest(BaseModel):
    graph_id: str
    title: str = "Research Knowledge Graph"
    height: str = "800px"
    width: str = "100%"
    node_size_factor: int = 10
    max_nodes: int = 300
    custom_colors: Optional[Dict[str, str]] = None

class VisualizationResponse(BaseModel):
    html_content: str
    graph_id: str
    title: str

class QueryRequest(BaseModel):
    query: str
    papers_ids: List[str]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

class ConversationRequest(BaseModel):
    paper_ids: List[str]

class ConversationResponse(BaseModel):
    conversation_id: str
    message: str = "Conversation created"

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]] = []

class HealthResponse(BaseModel):
    status: str = "healthy"
    api_version: str = "1.0.0"
    environment: str = "development"
    groq_api_configured: bool
