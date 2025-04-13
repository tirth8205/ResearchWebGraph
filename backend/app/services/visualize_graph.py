import networkx as nx
import logging
import os
from typing import Optional, Dict, Any
import importlib.util
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if pyvis is installed
if importlib.util.find_spec("pyvis") is None:
    logger.error("pyvis is not installed. Please install it with: pip install pyvis")
    raise ImportError("pyvis module is required for graph visualization")
else:
    from pyvis.network import Network

def visualize_graph(
    knowledge_graph: nx.Graph,
    output_path: str = "knowledge_graph.html",
    title: str = "Research Knowledge Graph",
    height: str = "800px",
    width: str = "100%",
    node_size_factor: int = 10,
    max_nodes: int = 500,
    custom_colors: Optional[Dict[str, str]] = None
) -> str:
    """
    Visualize the knowledge graph in an interactive HTML file.
    
    Args:
        knowledge_graph: NetworkX graph to visualize
        output_path: Path to save the HTML output
        title: Title for the visualization
        height: Height of the visualization
        width: Width of the visualization
        node_size_factor: Factor to multiply node weights for sizing
        max_nodes: Maximum number of nodes to include (to prevent browser crashes)
        custom_colors: Dictionary mapping node types to colors
        
    Returns:
        Path to the generated HTML file or error message
    """
    if not knowledge_graph:
        logger.warning("Empty graph provided for visualization")
        return "Empty graph, nothing to visualize"
    
    try:
        # Get graph statistics
        node_count = knowledge_graph.number_of_nodes()
        edge_count = knowledge_graph.number_of_edges()
        logger.info(f"Visualizing graph with {node_count} nodes and {edge_count} edges")
        
        # Check if graph is too large for visualization
        if node_count > max_nodes:
            logger.warning(f"Graph has {node_count} nodes, which exceeds the maximum of {max_nodes}")
            logger.info("Creating a subgraph with the most important nodes for visualization")
            
            # Get node centrality to identify important nodes
            try:
                centrality = nx.degree_centrality(knowledge_graph)
            except Exception:
                centrality = {node: knowledge_graph.degree(node) for node in knowledge_graph.nodes()}
            
            # Get top nodes by centrality
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            
            # Create subgraph with top nodes
            knowledge_graph = knowledge_graph.subgraph(top_node_ids)
            logger.info(f"Created subgraph with {knowledge_graph.number_of_nodes()} nodes")
        
        # Default node colors by type
        default_colors = {
            "DOCUMENT": "#3498db",  # Blue
            "PERSON": "#e74c3c",    # Red
            "ORG": "#2ecc71",       # Green
            "PRODUCT": "#f39c12",   # Orange
            "ACADEMIC_TERM": "#9b59b6",  # Purple
            "TECH_TERM": "#1abc9c",  # Teal
            "GPE": "#34495e",       # Dark blue
            "NORP": "#95a5a6",      # Gray
            "WORK_OF_ART": "#d35400",  # Dark orange
            "DATE": "#7f8c8d",      # Gray,
            "entity": "#3498db"     # Default blue for entities
        }
        
        colors = custom_colors if custom_colors else default_colors
        
        # Create pyvis network - Fixed duplicate title issue by removing heading parameter.
        is_directed = isinstance(knowledge_graph, nx.DiGraph)
        net = Network(
            height=height,
            width=width,
            directed=is_directed,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#000000"
        )
        
        net.barnes_hut(gravity=-80000, damping=0.09)

        net.from_nx(knowledge_graph)

        # Enhanced node customization with error handling.
        for node in net.nodes:
            try:
                if node['id'] in knowledge_graph.nodes:
                    node_data = knowledge_graph.nodes[node['id']]
                    entity_type = node_data.get('label', 'Unknown')
                    metadata = node_data.get('metadata', {})
                    
                    if entity_type == 'DOCUMENT':
                        title = metadata.get('title', 'Untitled')
                        authors = metadata.get('authors', [])
                        authors_str = ", ".join(authors[:3]) + (" and more..." if len(authors) > 3 else "")
                        node['title'] = f"<strong>{title}</strong><br>Authors: {authors_str}"
                        node['color'] = colors.get(entity_type, '#3498db')
                        node['shape'] = 'square'
                    else:
                        connection_count = knowledge_graph.degree(node['id'])
                        node['title'] = f"<strong>{node['id']}</strong><br>Type: {entity_type}<br>Connections: {connection_count}"
                        node['color'] = colors.get(entity_type, '#aaaaaa')
                        node['shape'] = 'dot'
                    
                    node['size'] = max(10, connection_count * 5)
                    
                else:
                    logger.warning(f"Node ID '{node['id']}' not found in graph data.")
                    node['title'] = f"Node ID '{node['id']}'"
                    node['color'] = '#aaaaaa'
                    node['shape'] = 'dot'
                    node['size'] = 10
            
            except Exception as e:
                logger.warning(f"Error processing node {node.get('id', 'unknown')}: {str(e)}")
                continue
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        net.save_graph(output_path)
        logger.info(f"Knowledge graph visualization saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        return f"Error during visualization: {str(e)}"
