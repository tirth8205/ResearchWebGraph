import os
import streamlit as st
import tempfile
import logging
from pathlib import Path
import importlib.util
import time
from typing import Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required packages
required_packages = [
    "streamlit", "arxiv", "langchain", "langchain_community", 
    "langchain_core", "langchain_huggingface", "faiss",  # Changed from faiss-cpu to faiss
    "networkx", "pyvis", "spacy"
]

missing_packages = []
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        missing_packages.append(package)

if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.info("Please install them with: pip install " + " ".join(missing_packages))
    st.stop()

# Import project modules
from fetch_papers import fetch_papers
from process_papers import process_papers
from build_graph import build_knowledge_graph
from visualize_graph import visualize_graph
from query_agent import create_agent

# Set page config
st.set_page_config(
    page_title="ResearchWebGraph",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "visualization_path" not in st.session_state:
    st.session_state.visualization_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = False

# Authentication and API key setup
def setup_api_keys():
    st.sidebar.header("API Key Setup")
    
    # HuggingFace Token
    hf_token = st.sidebar.text_input(
        "HuggingFace API Token",
        type="password",
        help="Required for embedding and LLM queries"
    )
    
    # Save button
    if st.sidebar.button("Save API Keys"):
        if hf_token:
            # Set environment variables
            os.environ["HF_TOKEN"] = hf_token
            st.session_state.api_keys_set = True
            st.sidebar.success("API keys saved successfully!")
            
            # Clear session state to force reload with new API keys
            st.session_state.documents = None
            st.session_state.vectorstore = None
            st.session_state.knowledge_graph = None
            st.session_state.agent = None
        else:
            st.sidebar.error("HuggingFace API Token is required")
    
    # Show status
    if st.session_state.api_keys_set:
        st.sidebar.info("✅ API keys are set")
    else:
        st.sidebar.warning("⚠️ API keys not configured")
    
    return st.session_state.api_keys_set

# Main application interface
def main():
    # Title and description
    st.title("📚 ResearchWebGraph")
    st.markdown("""
    Explore academic papers with knowledge graphs and AI assistance.
    
    This tool fetches research papers, builds a knowledge graph of concepts and 
    relationships, and lets you ask questions about the research.
    """)
    
    # Setup API keys in sidebar
    api_keys_ready = setup_api_keys()
    
    # Research Topic Input
    with st.expander("1. Fetch Research Papers", expanded=not st.session_state.documents):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Research Topic", 
                                 placeholder="e.g., quantum computing, climate change", 
                                 help="Enter a research topic to fetch papers from arXiv")
        with col2:
            max_papers = st.number_input("Max Papers", min_value=1, max_value=20, value=5)
        
        # Advanced options - Fixed: Changed from nested expander to checkbox
        show_advanced = st.checkbox("Show Advanced Options")
        if show_advanced:
            st.markdown("### Advanced Options")
            categories = st.multiselect(
                "arXiv Categories",
                options=["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML", "physics", "q-bio"],
                default=None,
                help="Filter by arXiv categories"
            )
            date_from = st.date_input("From Date", value=None, help="Filter papers published after this date")
            date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
        else:
            categories = None
            date_from_str = None
        
        # Fetch button
        fetch_col1, fetch_col2 = st.columns([1, 3])
        with fetch_col1:
            fetch_button = st.button("🔍 Fetch Papers", disabled=not api_keys_ready)
        
        # Process when button is clicked
        if fetch_button and query:
            try:
                with st.spinner(f"Fetching papers on '{query}'..."):
                    documents = fetch_papers(
                        query=query,
                        max_docs=max_papers,
                        categories=categories if categories else None,
                        date_from=date_from_str
                    )
                    
                    if not documents:
                        st.error("No papers found. Try a different query or check your connection.")
                    else:
                        st.session_state.documents = documents
                        st.success(f"Successfully fetched {len(documents)} papers!")
                        
                        # Show paper titles
                        st.subheader("Fetched Papers:")
                        for i, doc in enumerate(documents, 1):
                            metadata = doc.metadata
                            title = metadata.get("title", "Untitled")
                            authors = metadata.get("authors", [])
                            authors_str = ", ".join(authors[:3])
                            if len(authors) > 3:
                                authors_str += f" and {len(authors) - 3} more"
                            
                            st.markdown(f"**{i}. {title}**  \n*{authors_str}*")
                        
                        # Auto-process papers
                        with st.spinner("Processing papers..."):
                            st.session_state.vectorstore = process_papers(documents)
                            if st.session_state.vectorstore:
                                st.success("Papers processed and vector store created!")
                            else:
                                st.error("Error processing papers. Check your API keys.")
                        
                        # Auto-build knowledge graph
                        with st.spinner("Building knowledge graph..."):
                            st.session_state.knowledge_graph = build_knowledge_graph(documents)
                            node_count = st.session_state.knowledge_graph.number_of_nodes()
                            edge_count = st.session_state.knowledge_graph.number_of_edges()
                            st.success(f"Knowledge graph built with {node_count} nodes and {edge_count} edges!")
                        
                        # Auto-create agent
                        with st.spinner("Creating query agent..."):
                            st.session_state.agent = create_agent(
                                st.session_state.vectorstore,
                                st.session_state.knowledge_graph
                            )
                            st.success("Query agent created successfully!")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error in paper fetching: {str(e)}", exc_info=True)
    
    # Display Knowledge Graph
    if st.session_state.knowledge_graph:
        with st.expander("2. Knowledge Graph Visualization", expanded=not st.session_state.visualization_path):
            # Create two columns for buttons
            viz_col1, viz_col2 = st.columns([1, 1])
            
            with viz_col1:
                if st.button("🔄 Generate/Refresh Visualization"):
                    with st.spinner("Generating knowledge graph visualization..."):
                        # Create persistent directory for visualizations
                        viz_dir = os.path.join(os.getcwd(), "visualizations")
                        os.makedirs(viz_dir, exist_ok=True)
                        
                        # Generate unique filename with timestamp
                        output_path = os.path.join(viz_dir, f"knowledge_graph_{int(time.time())}.html")
                        
                        # Generate visualization
                        vis_path = visualize_graph(
                            st.session_state.knowledge_graph,
                            output_path=output_path
                        )
                        
                        # Save the path
                        st.session_state.visualization_path = vis_path
            
            # Display visualization if available
            if st.session_state.visualization_path and os.path.exists(st.session_state.visualization_path):
                # Check if file has content
                if os.path.getsize(st.session_state.visualization_path) > 0:
                    with open(st.session_state.visualization_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    # Display the visualization
                    st.components.v1.html(html_content, height=700)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Knowledge Graph HTML",
                        data=html_content,
                        file_name="knowledge_graph.html",
                        mime="text/html"
                    )
                else:
                    st.warning("Visualization file is empty. Please try generating it again.")
            elif st.session_state.visualization_path:
                st.warning("Visualization file not found. Please generate the visualization again.")
    
    # Query Interface
    if st.session_state.agent:
        with st.expander("3. Ask Questions About the Research", expanded=True):
            st.subheader("Research Assistant")
            
            # Display chat history
            chat_container = st.container(height=400)
            with chat_container:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**Assistant:** {message}")
            
            # Input for new questions
            user_input = st.text_input("Ask a question about the papers:", 
                                      key="user_question",
                                      placeholder="e.g., What are the main findings? Or, Show entities related to neural networks")
            
            if st.button("Ask", key="ask_button"):
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append(("user", user_input))
                    
                    # Get response from agent
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.agent.invoke({"input": user_input})
                            output = response.get("output", "I couldn't generate a response. Please try again.")
                            
                            # Add agent response to chat history
                            st.session_state.chat_history.append(("assistant", output))
                            
                            # Rerun to update the chat display - FIXED: Changed from experimental_rerun to rerun
                            st.rerun()
                            
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.session_state.chat_history.append(("assistant", error_msg))
                            logger.error(f"Error in agent response: {str(e)}", exc_info=True)
                            st.rerun()  # FIXED: Changed from experimental_rerun to rerun
    
    # Footer
    st.markdown("---")
    st.markdown(
        "ResearchWebGraph combines vector search, knowledge graphs, and LLMs to help you explore research papers."
    )

if __name__ == "__main__":
    main()
