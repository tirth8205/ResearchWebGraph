import streamlit as st
import os
import time
import networkx as nx
from typing import List, Dict, Any, Optional
import pandas as pd
import json

# Import components
from components.api import build_graph, get_visualization
from components.sidebar import show_graph_settings

# Page configuration
st.set_page_config(
    page_title="ResearchWebGraph - Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if API keys are set
if not st.session_state.get("api_keys_set", False):
    st.warning("⚠️ Please configure your API keys in the sidebar first.")
    st.stop()

# Page header
st.markdown('<h1 class="main-header">Knowledge Graph</h1>', unsafe_allow_html=True)

# Show graph settings in sidebar
show_graph_settings()

# First, check if papers are selected
selected_papers = st.session_state.get("selected_papers", [])

if not selected_papers:
    st.warning("⚠️ No papers selected. Please select papers in the Research Papers tab first.")
    
    # Add a navigation button
    if st.button("Go to Paper Selection"):
        from components.api import change_page
        change_page("01_Papers")
    
    st.stop()

# Display selected papers summary
st.markdown('<h2 class="sub-header">Selected Papers</h2>', unsafe_allow_html=True)

# Create a data frame for selected papers
paper_data = []
for paper in selected_papers:
    title = paper.get("metadata", {}).get("title", "Untitled")
    authors = ", ".join(paper.get("metadata", {}).get("authors", ["Unknown"]))
    # Truncate authors if too long
    if len(authors) > 50:
        authors = authors[:47] + "..."
    
    paper_data.append({
        "Title": title,
        "Authors": authors,
        "ID": paper.get("id", "unknown")
    })

# Show paper table
paper_df = pd.DataFrame(paper_data)
st.dataframe(paper_df, hide_index=True, use_container_width=True)

# Knowledge Graph Building Section
st.markdown('<h2 class="sub-header">Knowledge Graph Builder</h2>', unsafe_allow_html=True)

# Explanation of the knowledge graph
with st.expander("About Knowledge Graphs", expanded=False):
    st.markdown("""
    A knowledge graph represents entities (like people, organizations, concepts) and their relationships extracted from the papers.
    
    ### What's in the Graph?
    
    - **Documents**: The papers you've selected
    - **Entities**: People, organizations, locations, academic terms, etc. mentioned in the papers
    - **Relationships**: How different entities relate to each other
    
    ### How to Use It
    
    - **Build the graph** using the button below
    - **Explore connections** by dragging nodes and hovering over them
    - **Zoom and pan** to navigate larger graphs
    - **Search for specific entities** using the search box
    
    ### Graph Legend
    
    - 📄 **Blue Squares**: Documents (Papers)
    - 👤 **Red Diamonds**: People
    - 🏢 **Green Triangles**: Organizations
    - 🧠 **Purple Circles**: Academic Concepts
    """)

# Graph building options
col1, col2 = st.columns(2)

with col1:
    # Settings for graph generation
    graph_title = st.text_input(
        "Graph Title", 
        value=st.session_state.get("graph_settings", {}).get("title", "Research Knowledge Graph")
    )

with col2:
    # Button to build/refresh the graph
    button_label = "Refresh Knowledge Graph" if st.session_state.get("knowledge_graph") else "Build Knowledge Graph"
    
    if st.button(button_label, use_container_width=True):
        # Get paper IDs
        paper_ids = [paper["id"] for paper in selected_papers]
        
        # Build the graph
        graph = build_graph(paper_ids)
        
        if graph:
            st.session_state.knowledge_graph = graph
            st.success(f"Knowledge graph built with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges!")
        else:
            st.error("Failed to build knowledge graph. Please try again.")

# Graph visualization section
if st.session_state.get("knowledge_graph"):
    st.markdown('<h2 class="sub-header">Graph Visualization</h2>', unsafe_allow_html=True)
    
    # Graph statistics
    graph = st.session_state.knowledge_graph
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Count entity types
    entity_counts = {}
    for node in nodes:
        node_type = node.get("label", "Unknown")
        if node_type not in entity_counts:
            entity_counts[node_type] = 0
        entity_counts[node_type] += 1
    
    # Display statistics
    st.markdown("### Graph Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", len(nodes))
    
    with col2:
        st.metric("Total Edges", len(edges))
    
    with col3:
        st.metric("Papers", entity_counts.get("DOCUMENT", 0))
    
    # Display entity type counts
    st.markdown("### Entity Types")
    entity_cols = st.columns(5)
    
    entity_display = [
        ("PERSON", "People", "👤"),
        ("ORG", "Organizations", "🏢"),
        ("ACADEMIC_TERM", "Academic Terms", "🧠"),
        ("TECH_TERM", "Tech Terms", "💻"),
        ("GPE", "Locations", "🌍")
    ]
    
    for i, (entity_type, display_name, emoji) in enumerate(entity_display):
        with entity_cols[i % 5]:
            st.metric(f"{emoji} {display_name}", entity_counts.get(entity_type, 0))
    
    # Display graph visualization
    st.markdown("### Interactive Visualization")
    
    # Get graph ID from session state
    graph_id = st.session_state.get("current_graph_id")
    
    if graph_id:
        # Get visualization
        html_content = get_visualization(graph_id, graph_title)
        
        if html_content:
            # Set up a container with border styling
            st.markdown(
                """
                <style>
                .graph-container {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    margin-bottom: 20px;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            # Add search functionality above the graph
            st.text_input(
                "Search Entities", 
                key="graph_search",
                placeholder="Type to find entities in the graph...",
                help="Search for people, organizations, concepts, etc."
            )
            
            # Display the visualization
            st.components.v1.html(html_content, height=700, scrolling=False)
            
            # Add export options
            st.download_button(
                label="📥 Download Visualization",
                data=html_content,
                file_name=f"{graph_title.replace(' ', '_')}.html",
                mime="text/html",
                help="Download the graph visualization as an HTML file to view offline"
            )
        else:
            st.error("Failed to generate visualization. Please try building the graph again.")
    else:
        st.error("No graph ID found. Please try building the graph again.")
    
    # Entity explorer section
    st.markdown("### Entity Explorer")
    
    # Create a search box for entities
    search_term = st.text_input(
        "Find Entities", 
        placeholder="Type to search for people, organizations, concepts...",
        help="Enter a term to search for in the graph entities"
    )
    
    if search_term:
        # Filter nodes based on search
        search_lower = search_term.lower()
        matching_nodes = []
        
        for node in nodes:
            # Check node ID (which often contains the text)
            node_id = node.get("id", "").lower()
            node_metadata = node.get("metadata", {})
            node_type = node.get("label", "Unknown")
            
            # For document nodes, check title and authors
            if node_type == "DOCUMENT" and node_metadata:
                title = node_metadata.get("title", "").lower()
                authors = " ".join(node_metadata.get("authors", [])).lower()
                
                if search_lower in title or search_lower in authors:
                    matching_nodes.append(node)
            # For entity nodes, check the ID (which contains the entity text)
            elif search_lower in node_id:
                matching_nodes.append(node)
        
        if matching_nodes:
            st.success(f"Found {len(matching_nodes)} matching entities")
            
            # Show matching entities
            for node in matching_nodes:
                node_type = node.get("label", "Unknown")
                node_id = node.get("id", "")
                
                # Format the display based on node type
                if node_type == "DOCUMENT":
                    title = node.get("metadata", {}).get("title", "Untitled Document")
                    st.markdown(f"📄 **Document**: {title}")
                else:
                    # Extract the text part from the ID (e.g., "person_john_smith" -> "john smith")
                    entity_text = node_id.split("_", 1)[1] if "_" in node_id else node_id
                    entity_text = entity_text.replace("_", " ").title()
                    
                    # Icon based on entity type
                    icon = "👤" if node_type == "PERSON" else "🏢" if node_type == "ORG" else "🧠" if node_type == "ACADEMIC_TERM" else "📌"
                    
                    st.markdown(f"{icon} **{node_type}**: {entity_text}")
        else:
            st.info(f"No entities matching '{search_term}' found in the graph")
    
    # Graph export section
    with st.expander("Export Graph Data", expanded=False):
        st.markdown("""
        You can export the graph data in various formats for further analysis or visualization in other tools.
        """)
        
        # Export graph data as JSON
        graph_json = json.dumps(graph, indent=2)
        st.download_button(
            label="Download Graph JSON",
            data=graph_json,
            file_name="knowledge_graph.json",
            mime="application/json"
        )
        
        # Export nodes and edges as CSV
        # First, convert nodes to a more CSV-friendly format
        nodes_csv_data = []
        for node in nodes:
            node_data = {
                "id": node.get("id", ""),
                "type": node.get("label", ""),
                "entity_class": node.get("type", "")
            }
            
            # Add metadata if available
            metadata = node.get("metadata", {})
            if metadata:
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            node_data[f"metadata_{key}"] = value
            
            nodes_csv_data.append(node_data)
        
        # Convert edges to CSV-friendly format
        edges_csv_data = []
        for edge in edges:
            edge_data = {
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "relationship": edge.get("label", ""),
                "weight": edge.get("weight", 1.0)
            }
            edges_csv_data.append(edge_data)
        
        # Create DataFrames
        nodes_df = pd.DataFrame(nodes_csv_data)
        edges_df = pd.DataFrame(edges_csv_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download Nodes CSV",
                data=nodes_df.to_csv(index=False),
                file_name="graph_nodes.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="Download Edges CSV",
                data=edges_df.to_csv(index=False),
                file_name="graph_edges.csv",
                mime="text/csv"
            )

else:
    # Show info message if no graph is built yet
    st.info("Click the 'Build Knowledge Graph' button above to generate a visualization of the research papers and their connections.")

# Footer
st.markdown("---")
st.markdown(
    "The knowledge graph extracts entities and relationships from the research papers, helping you discover connections and insights that might not be immediately obvious."
)
