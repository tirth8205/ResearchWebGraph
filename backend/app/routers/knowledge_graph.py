from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Optional, Dict, Any
import logging
import os
import json
from datetime import datetime
import uuid

# Import models and services
from app.models.schemas import KnowledgeGraph, GraphVisualizationRequest, GraphVisualizationResponse
from app.services.build_graph import build_knowledge_graph
from app.services.visualize_graph import visualize_graph

# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# In-memory storage for knowledge graphs
# In a production environment, consider using a database
graph_storage = {}

@router.post("/build", response_model=KnowledgeGraph)
async def create_knowledge_graph(
    paper_ids: List[str] = Body(..., description="List of paper IDs to include in the graph")
):
    """
    Build a knowledge graph from processed papers.
    
    Creates a knowledge graph by extracting entities and relationships from the specified papers.
    The graph can be used for visualization and querying connections between concepts.
    """
    try:
        if not paper_ids:
            raise HTTPException(status_code=400, detail="At least one paper ID must be provided")
        
        logger.info(f"Building knowledge graph from {len(paper_ids)} papers")
        
        # Build the knowledge graph
        graph = await build_knowledge_graph(paper_ids)
        
        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())
        
        # Store the graph for later retrieval
        graph_storage[graph_id] = {
            "graph": graph,
            "paper_ids": paper_ids,
            "created_at": datetime.now().isoformat(),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges)
        }
        
        logger.info(f"Successfully built knowledge graph {graph_id} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Add graph_id to the response
        graph_dict = graph.dict()
        graph_dict["id"] = graph_id
        
        return graph_dict
    
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error building knowledge graph: {str(e)}")

@router.get("/list")
async def list_knowledge_graphs(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    List available knowledge graphs.
    
    Returns a list of knowledge graphs that have been created, with information about each graph.
    """
    try:
        # Get all graph IDs and sort by creation time (newest first)
        graph_ids = list(graph_storage.keys())
        graph_ids.sort(key=lambda x: graph_storage[x]["created_at"], reverse=True)
        
        # Apply pagination
        paginated_ids = graph_ids[offset:offset+limit]
        
        # Get graph summaries
        graphs = []
        for graph_id in paginated_ids:
            graph_info = graph_storage[graph_id]
            graphs.append({
                "id": graph_id,
                "paper_ids": graph_info["paper_ids"],
                "created_at": graph_info["created_at"],
                "node_count": graph_info["node_count"],
                "edge_count": graph_info["edge_count"]
            })
        
        return {
            "graphs": graphs,
            "total": len(graph_ids),
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        logger.error(f"Error listing knowledge graphs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing knowledge graphs: {str(e)}")

@router.get("/{graph_id}", response_model=KnowledgeGraph)
async def get_knowledge_graph(graph_id: str):
    """
    Retrieve a specific knowledge graph by ID.
    
    Gets the complete knowledge graph data including all nodes and edges.
    """
    try:
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Knowledge graph {graph_id} not found")
        
        graph = graph_storage[graph_id]["graph"]
        
        # Add graph_id to the response
        graph_dict = graph.dict()
        graph_dict["id"] = graph_id
        
        return graph_dict
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving knowledge graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving knowledge graph: {str(e)}")

@router.post("/visualize", response_model=GraphVisualizationResponse)
async def get_visualization(request: GraphVisualizationRequest):
    """
    Generate a visualization of a knowledge graph.
    
    Creates an interactive HTML visualization of the specified knowledge graph.
    """
    try:
        graph_id = request.graph_id
        
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Knowledge graph {graph_id} not found")
        
        # Get the graph
        graph = graph_storage[graph_id]["graph"]
        
        # Generate visualization
        html_content = await visualize_graph(
            graph, 
            title=request.title,
            height=request.height,
            width=request.width
        )
        
        logger.info(f"Generated visualization for knowledge graph {graph_id}")
        
        return GraphVisualizationResponse(
            html_content=html_content,
            graph_id=graph_id
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@router.delete("/{graph_id}")
async def delete_knowledge_graph(graph_id: str):
    """
    Delete a knowledge graph.
    
    Removes a knowledge graph from the system.
    """
    try:
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Knowledge graph {graph_id} not found")
        
        # Remove the graph
        del graph_storage[graph_id]
        
        logger.info(f"Deleted knowledge graph {graph_id}")
        
        return {"message": f"Knowledge graph {graph_id} deleted successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error deleting knowledge graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting knowledge graph: {str(e)}")

@router.get("/search/entities")
async def search_entities(
    query: str = Query(..., min_length=1),
    graph_id: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
):
    """
    Search for entities in knowledge graphs.
    
    Finds entities matching the query string in either a specific graph or across all graphs.
    """
    try:
        results = []
        
        # Determine which graphs to search
        graphs_to_search = []
        if graph_id:
            if graph_id not in graph_storage:
                raise HTTPException(status_code=404, detail=f"Knowledge graph {graph_id} not found")
            graphs_to_search = [(graph_id, graph_storage[graph_id]["graph"])]
        else:
            graphs_to_search = [(gid, graph_storage[gid]["graph"]) for gid in graph_storage]
        
        # Search for matching entities
        query_lower = query.lower()
        for gid, graph in graphs_to_search:
            for node in graph.nodes:
                # Check if query matches node id or any metadata
                if query_lower in node.id.lower():
                    results.append({
                        "entity_id": node.id,
                        "entity_type": node.type,
                        "entity_label": node.label,
                        "graph_id": gid
                    })
                    if len(results) >= limit:
                        break
            
            if len(results) >= limit:
                break
        
        return {
            "entities": results,
            "count": len(results),
            "query": query
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching entities: {str(e)}")

@router.get("/relations/{entity_id}")
async def get_entity_relations(
    entity_id: str,
    graph_id: str,
    direction: str = Query("both", enum=["incoming", "outgoing", "both"])
):
    """
    Get relations for a specific entity.
    
    Returns all relationships connected to the specified entity.
    """
    try:
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Knowledge graph {graph_id} not found")
        
        graph = graph_storage[graph_id]["graph"]
        
        # Check if entity exists
        entity_exists = False
        for node in graph.nodes:
            if node.id == entity_id:
                entity_exists = True
                break
        
        if not entity_exists:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found in graph {graph_id}")
        
        # Find relationships
        incoming = []
        outgoing = []
        
        if direction in ["incoming", "both"]:
            # Find incoming relationships (where entity is the target)
            for edge in graph.edges:
                if edge.target == entity_id:
                    # Find the source node details
                    source_node = next((node for node in graph.nodes if node.id == edge.source), None)
                    if source_node:
                        incoming.append({
                            "source_id": edge.source,
                            "source_type": source_node.type,
                            "source_label": source_node.label,
                            "relation": edge.label,
                            "weight": edge.weight
                        })
        
        if direction in ["outgoing", "both"]:
            # Find outgoing relationships (where entity is the source)
            for edge in graph.edges:
                if edge.source == entity_id:
                    # Find the target node details
                    target_node = next((node for node in graph.nodes if node.id == edge.target), None)
                    if target_node:
                        outgoing.append({
                            "target_id": edge.target,
                            "target_type": target_node.type,
                            "target_label": target_node.label,
                            "relation": edge.label,
                            "weight": edge.weight
                        })
        
        return {
            "entity_id": entity_id,
            "incoming_relations": incoming,
            "outgoing_relations": outgoing,
            "total_relations": len(incoming) + len(outgoing)
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error getting entity relations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting entity relations: {str(e)}")
