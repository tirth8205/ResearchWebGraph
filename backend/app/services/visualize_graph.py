import os
import logging
import asyncio
import networkx as nx
from typing import Dict, Any, Optional, Union
import uuid
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Import models
from app.models.schemas import KnowledgeGraph

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import visualization library
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    logger.warning("PyVis not available. Installing required packages...")
    try:
        import subprocess
        subprocess.run(["pip", "install", "pyvis"], check=True)
        from pyvis.network import Network
        PYVIS_AVAILABLE = True
        logger.info("Successfully installed PyVis")
    except Exception as e:
        logger.error(f"Failed to install PyVis: {str(e)}")
        PYVIS_AVAILABLE = False

# Directory for visualization output
VISUALIZATION_DIR = os.getenv("VISUALIZATION_DIR", "visualizations")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

async def visualize_graph(
    graph: Union[KnowledgeGraph, Dict[str, Any]],
    title: str = "Research Knowledge Graph",
    height: str = "800px",
    width: str = "100%",
    node_size_factor: int = 10,
    max_nodes: int = 300,
    custom_colors: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate an HTML visualization of a knowledge graph.
    
    Args:
        graph: KnowledgeGraph to visualize
        title: Title for the visualization
        height: Height of the visualization
        width: Width of the visualization
        node_size_factor: Factor to multiply node weights for sizing
        max_nodes: Maximum number of nodes to include
        custom_colors: Optional dictionary mapping node types to colors
        
    Returns:
        HTML content as a string
    """
    # Run visualization in a separate thread to avoid blocking
    return await asyncio.to_thread(
        _visualize_graph_sync,
        graph,
        title,
        height,
        width,
        node_size_factor,
        max_nodes,
        custom_colors
    )

def _visualize_graph_sync(
    graph: Union[KnowledgeGraph, Dict[str, Any]],
    title: str = "Research Knowledge Graph",
    height: str = "800px",
    width: str = "100%",
    node_size_factor: int = 10,
    max_nodes: int = 300,
    custom_colors: Optional[Dict[str, str]] = None
) -> str:
    """
    Synchronous implementation of visualize_graph.
    
    Args:
        graph: KnowledgeGraph to visualize
        title: Title for the visualization
        height: Height of the visualization
        width: Width of the visualization
        node_size_factor: Factor to multiply node weights for sizing
        max_nodes: Maximum number of nodes to include
        custom_colors: Optional dictionary mapping node types to colors
        
    Returns:
        HTML content as a string
    """
    try:
        if not PYVIS_AVAILABLE:
            return _generate_fallback_visualization(graph, title)
        
        # Convert dict to KnowledgeGraph if needed
        if isinstance(graph, dict):
            graph = KnowledgeGraph(**graph)
        
        # Get nodes and edges from graph
        nodes = graph.nodes
        edges = graph.edges
        
        if not nodes:
            logger.warning("Empty graph (no nodes) provided for visualization")
            return "<html><body><h1>Empty Graph</h1><p>No nodes to visualize.</p></body></html>"
        
        logger.info(f"Visualizing graph with {len(nodes)} nodes and {len(edges)} edges")
        
        # Check if graph is too large and limit size if needed
        if len(nodes) > max_nodes:
            logger.warning(f"Graph has {len(nodes)} nodes, which exceeds the maximum of {max_nodes}")
            nodes, edges = _reduce_graph_size(nodes, edges, max_nodes)
            logger.info(f"Reduced graph to {len(nodes)} nodes and {len(edges)} edges")
        
        # Create NetworkX graph for visualization
        G = nx.DiGraph()
        
        # Add nodes to NetworkX graph
        for node in nodes:
            node_data = {
                "id": node.id,
                "label": node.label,
                "type": node.type
            }
            
            if node.metadata:
                for key, value in node.metadata.items():
                    # Skip complex structures that might cause issues
                    if isinstance(value, (str, int, float, bool)):
                        node_data[key] = value
            
            G.add_node(node.id, **node_data)
        
        # Add edges to NetworkX graph
        for edge in edges:
            G.add_edge(
                edge.source,
                edge.target,
                label=edge.label,
                weight=edge.weight
            )
        
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
            "DATE": "#7f8c8d",      # Gray
            "entity": "#3498db"     # Default blue for entities
        }
        
        # Use custom colors if provided
        colors = custom_colors if custom_colors else default_colors
        
        # Create pyvis network - FIXED: Removed heading to prevent duplicate title
        is_directed = True  # Knowledge graphs are typically directed
        net = Network(
            height=height,
            width=width,
            directed=is_directed,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#000000"  # No heading parameter to avoid duplicate title
        )
        
        # Configure physics for better visualization
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.001,
            damping=0.09,
            overlap=0
        )
        
        # Add the graph to pyvis
        net.from_nx(G)
        
        # Customize nodes
        for node in net.nodes:
            try:
                if 'id' in node and node['id'] in G.nodes:
                    node_data = G.nodes[node['id']]
                    node_type = node_data.get('type', 'entity')
                    node_label = node_data.get('label', 'Unknown')
                    
                    # Format title (hover text) with all attributes
                    hover_info = []
                    if node_type == 'document':
                        # For document nodes, show title and authors
                        document_title = node_data.get('title', node_data.get('id', 'Untitled'))
                        authors = node_data.get('authors', [])
                        if isinstance(authors, list) and authors:
                            authors_text = ", ".join(authors[:3])
                            if len(authors) > 3:
                                authors_text += f" and {len(authors) - 3} more"
                        else:
                            authors_text = "Unknown authors"
                        
                        hover_info.append(f"<strong>{document_title}</strong>")
                        hover_info.append(f"Authors: {authors_text}")
                        published = node_data.get('published', 'Unknown date')
                        hover_info.append(f"Published: {published}")
                        
                        # Use a more readable label for the node
                        if len(document_title) > 30:
                            node['label'] = document_title[:27] + "..."
                        else:
                            node['label'] = document_title
                    else:
                        # For entity nodes, show type and text
                        entity_text = node_data.get('text', node['id'])
                        hover_info.append(f"<strong>{entity_text}</strong>")
                        hover_info.append(f"Type: {node_label}")
                        count = node_data.get('count', 1)
                        hover_info.append(f"Occurrences: {count}")
                        
                        # Use entity text as label
                        if len(entity_text) > 20:
                            node['label'] = entity_text[:17] + "..."
                        else:
                            node['label'] = entity_text
                    
                    # Set node HTML title (hover text)
                    node['title'] = "<br>".join(hover_info)
                    
                    # Set node color based on type
                    node['color'] = colors.get(node_label, colors.get('entity', '#3498db'))
                    
                    # Set node shape based on type
                    if node_type == 'document':
                        node['shape'] = 'square'
                        node['size'] = 30  # Larger size for documents
                    elif node_label == 'PERSON':
                        node['shape'] = 'diamond'
                    elif node_label == 'ORG':
                        node['shape'] = 'triangle'
                    elif node_label == 'ACADEMIC_TERM':
                        node['shape'] = 'dot'
                    else:
                        node['shape'] = 'dot'
                    
                    # Set font size based on importance
                    if node_type == 'document':
                        node['font'] = {'size': 14}
                    else:
                        node['font'] = {'size': 12}
                    
                    # Set node size based on importance or occurrences
                    if 'size' not in node:
                        if node_type == 'document':
                            node['size'] = 30
                        else:
                            count = node_data.get('count', 1)
                            node['size'] = count * node_size_factor + 10
            except Exception as e:
                logger.warning(f"Error customizing node {node.get('id', 'unknown')}: {str(e)}")
                # Apply default styling
                node['color'] = '#aaaaaa'
                node['shape'] = 'dot'
                node['size'] = 10
        
        # Customize edges
        for edge in net.edges:
            try:
                if 'from' in edge and 'to' in edge:
                    # Set edge label to be visible
                    if 'label' in edge:
                        # Ensure label is not too long
                        if len(edge['label']) > 15:
                            edge['label'] = edge['label'][:12] + "..."
                    
                    # Set arrow direction
                    if is_directed:
                        edge['arrows'] = 'to'
                    
                    # Set edge width based on weight
                    if 'weight' in edge:
                        weight = float(edge['weight'])
                        # Scale weight to reasonable width
                        edge['width'] = min(5, max(1, weight))
                    else:
                        edge['width'] = 1
            except Exception as e:
                logger.warning(f"Error customizing edge: {str(e)}")
        
        # Add custom HTML to create a single title
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .graph-title {{
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 20px;
                }}
                .controls {{
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .legend {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-top: 10px;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin-right: 15px;
                    margin-bottom: 5px;
                }}
                .legend-color {{
                    width: 15px;
                    height: 15px;
                    border-radius: 3px;
                    margin-right: 5px;
                }}
                #mynetwork {{
                    width: {width};
                    height: {height};
                    border: 1px solid lightgray;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="graph-title">{title}</div>
            <div class="controls">
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #3498db;"></div>
                        <span>Document</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #e74c3c;"></div>
                        <span>Person</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #2ecc71;"></div>
                        <span>Organization</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #9b59b6;"></div>
                        <span>Academic Term</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #1abc9c;"></div>
                        <span>Technical Term</span>
                    </div>
                </div>
            </div>
            <div id="mynetwork"></div>
        """
        
        # Replace the default HTML with our custom template
        if hasattr(net, 'html'):
            body_content = net.html.split('<body>')[1] if '<body>' in net.html else net.html
            net.html = html_template + body_content
        
        # Generate a unique filename for the visualization
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(VISUALIZATION_DIR, f"graph_{timestamp}_{random_id}.html")
        
        # Save the visualization
        net.save_graph(output_path)
        logger.info(f"Saved visualization to {output_path}")
        
        # Read the HTML content
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return html_content
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        return _generate_error_visualization(str(e))

def _reduce_graph_size(nodes, edges, max_nodes):
    """
    Reduce the size of a graph by keeping the most important nodes.
    
    Args:
        nodes: List of graph nodes
        edges: List of graph edges
        max_nodes: Maximum number of nodes to keep
        
    Returns:
        Tuple of (filtered_nodes, filtered_edges)
    """
    # Calculate node importance based on connectivity
    node_importance = {}
    
    # Count connections for each node
    for edge in edges:
        if edge.source not in node_importance:
            node_importance[edge.source] = 0
        if edge.target not in node_importance:
            node_importance[edge.target] = 0
        
        node_importance[edge.source] += 1
        node_importance[edge.target] += 1
    
    # Ensure document nodes are always included
    doc_nodes = [node for node in nodes if node.type == "document"]
    doc_node_ids = {node.id for node in doc_nodes}
    
    # Get top entity nodes by importance
    entity_nodes = [node for node in nodes if node.id not in doc_node_ids]
    
    # Sort entity nodes by importance
    sorted_entity_nodes = sorted(
        entity_nodes,
        key=lambda node: node_importance.get(node.id, 0),
        reverse=True
    )
    
    # Keep top N entity nodes
    keep_entity_count = max_nodes - len(doc_nodes)
    if keep_entity_count < 0:
        keep_entity_count = 0
    
    top_entity_nodes = sorted_entity_nodes[:keep_entity_count]
    
    # Combine document nodes and top entity nodes
    filtered_nodes = doc_nodes + top_entity_nodes
    
    # Filter edges to only include edges between kept nodes
    kept_node_ids = {node.id for node in filtered_nodes}
    filtered_edges = [
        edge for edge in edges 
        if edge.source in kept_node_ids and edge.target in kept_node_ids
    ]
    
    return filtered_nodes, filtered_edges

def _generate_fallback_visualization(graph, title):
    """
    Generate a simple HTML table visualization if PyVis is not available.
    
    Args:
        graph: Knowledge graph
        title: Title for the visualization
        
    Returns:
        HTML content as string
    """
    try:
        if isinstance(graph, dict):
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
        else:
            nodes = graph.nodes
            edges = graph.edges
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>This is a fallback visualization. For interactive graphs, please install PyVis.</p>
            
            <h2>Nodes ({len(nodes)})</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Label</th>
                </tr>
        """
        
        # Add node rows
        for node in nodes:
            node_id = node.id if hasattr(node, "id") else node.get("id", "")
            node_type = node.type if hasattr(node, "type") else node.get("type", "")
            node_label = node.label if hasattr(node, "label") else node.get("label", "")
            
            html += f"""
                <tr>
                    <td>{node_id}</td>
                    <td>{node_type}</td>
                    <td>{node_label}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Edges</h2>
            <table>
                <tr>
                    <th>Source</th>
                    <th>Target</th>
                    <th>Relationship</th>
                    <th>Weight</th>
                </tr>
        """
        
        # Add edge rows
        for edge in edges:
            source = edge.source if hasattr(edge, "source") else edge.get("source", "")
            target = edge.target if hasattr(edge, "target") else edge.get("target", "")
            label = edge.label if hasattr(edge, "label") else edge.get("label", "")
            weight = edge.weight if hasattr(edge, "weight") else edge.get("weight", "")
            
            html += f"""
                <tr>
                    <td>{source}</td>
                    <td>{target}</td>
                    <td>{label}</td>
                    <td>{weight}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    except Exception as e:
        logger.error(f"Error generating fallback visualization: {str(e)}", exc_info=True)
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"

def _generate_error_visualization(error_message):
    """
    Generate an error visualization HTML page.
    
    Args:
        error_message: Error message to display
        
    Returns:
        HTML content as string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Visualization Error</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                color: #333;
                line-height: 1.6;
            }}
            .error-container {{
                border: 1px solid #e74c3c;
                border-radius: 5px;
                padding: 20px;
                margin-top: 20px;
                background-color: #fadbd8;
            }}
            h1 {{ color: #e74c3c; }}
            pre {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                overflow: auto;
            }}
        </style>
    </head>
    <body>
        <h1>Visualization Error</h1>
        <p>An error occurred while generating the knowledge graph visualization:</p>
        
        <div class="error-container">
            <pre>{error_message}</pre>
        </div>
        
        <p>Please check the application logs for more details.</p>
    </body>
    </html>
    """
    
    return html
