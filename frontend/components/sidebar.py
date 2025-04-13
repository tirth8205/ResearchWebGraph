import streamlit as st
import os
from dotenv import set_key, load_dotenv
import requests
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load existing environment variables from .env
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)

def setup_api_keys():
    """Setup API keys and backend configuration in the sidebar."""
    with st.sidebar.expander("API Configuration", expanded=not st.session_state.get("api_keys_set", False)):
        # Groq API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Required for LLM access. Get one at https://console.groq.com/"
        )
        
        # Backend URL input
        backend_url = st.text_input(
            "Backend URL",
            value=os.getenv("BACKEND_URL", "http://localhost:8000"),
            help="URL of the FastAPI backend server"
        )
        
        # Qdrant Configuration
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=os.getenv("QDRANT_URL", "http://localhost:6333"),
            help="URL of the Qdrant vector database"
        )
        
        qdrant_collection_name = st.text_input(
            "Qdrant Collection Name",
            value=os.getenv("QDRANT_COLLECTION_NAME", "research_papers"),
            help="Name of the Qdrant collection for storing embeddings"
        )
        
        # Embedding Model
        embedding_model = st.text_input(
            "Embedding Model",
            value=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            help="SentenceTransformer model for generating embeddings"
        )
        
        # Chunk Settings
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=100,
            max_value=5000,
            value=int(os.getenv("CHUNK_SIZE", 1000)),
            help="Size of text chunks when splitting documents"
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=1000,
            value=int(os.getenv("CHUNK_OVERLAP", 200)),
            help="Overlap between consecutive text chunks"
        )
        
        # Save settings button
        if st.button("Save Configuration"):
            # Update environment variables
            set_key(dotenv_path, "GROQ_API_KEY", groq_api_key)
            set_key(dotenv_path, "BACKEND_URL", backend_url)
            set_key(dotenv_path, "QDRANT_URL", qdrant_url)
            set_key(dotenv_path, "QDRANT_COLLECTION_NAME", qdrant_collection_name)
            set_key(dotenv_path, "EMBEDDING_MODEL", embedding_model)
            set_key(dotenv_path, "CHUNK_SIZE", str(chunk_size))
            set_key(dotenv_path, "CHUNK_OVERLAP", str(chunk_overlap))
            
            # Update session state
            st.session_state.api_keys_set = True
            st.session_state.backend_url = backend_url
            
            # Success message
            st.sidebar.success("✅ Configuration saved successfully!")
    
    # Show system status indicators
    if st.session_state.get("api_keys_set", False):
        st.sidebar.success("✅ API Keys: Configured")
    else:
        st.sidebar.warning("⚠️ API Keys: Not configured")
    
    if backend_url:
        try:
            response = requests.get(f"{backend_url}/health", timeout=2)
            
            if response.status_code == 200:
                st.sidebar.success("✅ Backend: Connected")
                return True
            else:
                st.sidebar.error("❌ Backend: Error")
                return False
        except Exception:
            st.sidebar.error("❌ Backend: Not connected")
    
    return False

def display_selected_papers(papers):
    """Display selected papers in the sidebar."""
    with st.sidebar.expander("Selected Papers", expanded=True):
        if not papers:
            st.info("No papers selected yet.")
            return
        
        # Show count and list of selected papers
        st.write(f"**{len(papers)} papers selected:**")
        
        for i, paper in enumerate(papers):
            title = paper.get("metadata", {}).get("title", f"Paper {i+1}")
            
            # Truncate long titles for display
            if len(title) > 40:
                display_title = title[:37] + "..."
            else:
                display_title = title
            
            paper_container = st.container()
            
            with paper_container:
                col1, col2 = st.columns([0.8, 0.2])
                
                with col1:
                    st.write(f"{i+1}. {display_title}")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{paper['id']}"):
                        # Remove paper from selected papers
                        st.session_state.selected_papers = [p for p in papers if p['id'] != paper['id']]
                        st.experimental_rerun()
        
        # Add buttons for actions on selected papers
        if len(papers) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Build Graph"):
                    from components.api import change_page
                    change_page("02_Knowledge_Graph")
            
            with col2:
                if st.button("🤖 Ask Questions"):
                    from components.api import change_page
                    change_page("03_Query_Assistant")
            
