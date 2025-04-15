import streamlit as st
import os
from typing import List, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

def setup_api_keys():
    """Setup API keys in the sidebar."""
    with st.sidebar.expander("API Configuration", expanded=not st.session_state.get("api_keys_set", False)):
        # Groq API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Required for LLM access. Get one at https://console.groq.com/"
        )
        
        # Add Semantic Scholar API Key input
        semantic_scholar_api_key = st.text_input(
            "Semantic Scholar API Key (Optional)",
            type="password",
            value=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
            help="Optional: For higher rate limits with Semantic Scholar API"
        )
        
        # Backend URL input
        backend_url = st.text_input(
            "Backend URL",
            value=st.session_state.get("backend_url", os.getenv("BACKEND_URL", "http://localhost:8000")),
            help="URL of the FastAPI backend server"
        )
        
        # Save settings button
        if st.button("Save Configuration"):
            if groq_api_key:
                # Update environment variables (for current session)
                os.environ["GROQ_API_KEY"] = groq_api_key
                os.environ["BACKEND_URL"] = backend_url
                
                # Save Semantic Scholar API key if provided
                if semantic_scholar_api_key:
                    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = semantic_scholar_api_key
                
                # Save to session state
                st.session_state.api_keys_set = True
                st.session_state.backend_url = backend_url
                
                # Log success
                logger.info("API configuration saved")
                st.sidebar.success("✅ Configuration saved successfully!")
            else:
                st.sidebar.error("❌ Groq API Key is required")
    
    # System status indicators
    if st.session_state.get("api_keys_set", False):
        st.sidebar.success("✅ API Keys: Configured")
    else:
        st.sidebar.warning("⚠️ API Keys: Not configured")
    
    # Check backend connection
    if st.session_state.get("backend_url"):
        try:
            import requests
            response = requests.get(f"{st.session_state.backend_url}/health", timeout=2)
            if response.status_code == 200:
                st.sidebar.success("✅ Backend: Connected")
            else:
                st.sidebar.error("❌ Backend: Error")
        except:
            st.sidebar.error("❌ Backend: Not connected")

def display_selected_papers(papers: List[Dict[str, Any]]):
    """Display selected papers in the sidebar."""
    with st.sidebar.expander("Selected Papers", expanded=True):
        if not papers:
            st.info("No papers selected yet")
            return
        
        # Show count and list of selected papers
        st.write(f"**{len(papers)} papers selected:**")
        
        for i, paper in enumerate(papers):
            # Get paper title with fallback
            title = paper.get("metadata", {}).get("title", f"Paper {i+1}")
            
            # Truncate long titles for display
            if len(title) > 40:
                display_title = title[:37] + "..."
            else:
                display_title = title
            
            # Create a container for each paper
            paper_container = st.container()
            
            with paper_container:
                col1, col2 = st.columns([0.8, 0.2])
                
                with col1:
                    st.write(f"{i+1}. {display_title}")
                
                with col2:
                    # Remove button (align to right)
                    if st.button("🗑️", key=f"remove_{paper['id']}", help=f"Remove {title}"):
                        # Remove paper from selected papers
                        st.session_state.selected_papers = [p for p in st.session_state.selected_papers if p['id'] != paper['id']]
                        st.rerun()
        
        # Add buttons for actions on selected papers
        if len(papers) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Build Graph", key="build_graph_sidebar"):
                    # Set navigation to the Knowledge Graph page
                    from components.api import change_page
                    change_page("02_Knowledge_Graph")
            
            with col2:
                if st.button("🤖 Ask Questions", key="ask_questions_sidebar"):
                    # Set navigation to the Query Assistant page
                    from components.api import change_page
                    change_page("03_Query_Assistant")
            
            # Clear selection button
            if st.button("Clear Selection", key="clear_selection"):
                st.session_state.selected_papers = []
                st.rerun()

def show_app_info():
    """Display application information in the sidebar."""
    with st.sidebar.expander("About ResearchWebGraph", expanded=False):
        st.markdown("""
        **ResearchWebGraph** is an AI-powered research assistant that helps you explore and understand academic papers.
        
        **Features:**
        - Search and analyze research papers
        - Build interactive knowledge graphs
        - Get AI-generated answers to questions
        
        **Technologies:**
        - Backend: FastAPI, Qdrant, SentenceTransformers
        - Frontend: Streamlit
        - NLP: NLTK, spaCy
        - LLM: Groq-powered language models
        
        **Version:** 1.0.0
        """)
        
        # Add link to documentation or GitHub repository
        st.markdown("[Documentation](https://github.com/yourusername/ResearchWebGraph)")

def show_paper_filters():
    """Show filters for paper search in the sidebar."""
    with st.sidebar.expander("Search Filters", expanded=False):
        # Category selection
        categories = st.multiselect(
            "Categories",
            options=[
                "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE",
                "stat.ML", "physics", "q-bio", "math.ST"
            ],
            default=[],
            help="Filter papers by arXiv categories"
        )
        
        # Date filter
        date_from = st.date_input(
            "From Date",
            value=None,
            help="Show papers published after this date"
        )
        
        # Store filters in session state
        if categories:
            st.session_state.search_categories = categories
        else:
            st.session_state.search_categories = None
            
        if date_from:
            st.session_state.search_date_from = date_from.strftime("%Y-%m-%d")
        else:
            st.session_state.search_date_from = None
        
        # Apply filters button
        if st.button("Apply Filters"):
            st.success("Filters applied!")
            # Navigation will use these filters from session state

def show_graph_settings():
    """Show settings for knowledge graph visualization in the sidebar."""
    with st.sidebar.expander("Graph Settings", expanded=False):
        # Node size factor
        node_size_factor = st.slider(
            "Node Size Factor",
            min_value=1,
            max_value=20,
            value=10,
            help="Multiplier for node sizes based on importance"
        )
        
        # Max nodes
        max_nodes = st.slider(
            "Maximum Nodes",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Maximum number of nodes to display for performance"
        )
        
        # Graph title
        graph_title = st.text_input(
            "Graph Title",
            value="Research Knowledge Graph",
            help="Title for the visualization"
        )
        
        # Store settings in session state
        st.session_state.graph_settings = {
            "node_size_factor": node_size_factor,
            "max_nodes": max_nodes,
            "title": graph_title
        }
        
        # Apply settings button
        if st.button("Apply Settings"):
            st.success("Settings applied!")
            # These will be used when generating the visualization
