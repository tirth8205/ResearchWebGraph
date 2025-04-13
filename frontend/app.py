import streamlit as st
from streamlit_option_menu import option_menu
import os
import logging
import time
from dotenv import load_dotenv

# Import components
from components.sidebar import setup_api_keys, display_selected_papers

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ResearchWebGraph",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 1px solid #FCD34D;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #3B82F6;
    }
    /* Hide "Made with Streamlit" footer */
    footer {visibility: hidden;}
    /* Hide hamburger menu */
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = bool(os.getenv("GROQ_API_KEY"))

if "backend_url" not in st.session_state:
    st.session_state.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

if "papers" not in st.session_state:
    st.session_state.papers = []

if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []

if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_graph_id" not in st.session_state:
    st.session_state.current_graph_id = None

# Sidebar for configuration and navigation
with st.sidebar:
    # Show app title
    st.title("🧠 ResearchWebGraph")
    
    # Setup API keys
    setup_api_keys()
    
    # Display selected papers
    if st.session_state.selected_papers:
        display_selected_papers(st.session_state.selected_papers)
    
    # Navigation menu for mobile/narrow screens
    st.markdown("---")
    st.markdown("### Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Research Papers", "Knowledge Graph", "Query Assistant"],
        icons=["house", "journal-text", "diagram-3", "chat-dots"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )
    
    # Version information
    st.markdown("---")
    st.markdown("v1.0.0")

# Main content based on selection
if selected == "Home":
    st.markdown('<h1 class="main-header">Welcome to ResearchWebGraph</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ResearchWebGraph is a powerful tool for exploring academic research using advanced AI and knowledge graph technologies.
    
    ## How it Works
    
    1. **Search and retrieve papers** from arXiv based on your research topics
    2. **Build knowledge graphs** to visualize connections between entities, concepts, and papers
    3. **Ask questions about the research** using the AI-powered query assistant
    
    This application combines vector search, knowledge graph analysis, and large language models to help you explore and understand research papers.
    """)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Paper Search")
        st.markdown("""
        - Search arXiv for research papers
        - Upload your own PDF documents
        - Process and analyze paper content
        """)
    
    with col2:
        st.markdown("### 🕸️ Knowledge Graph")
        st.markdown("""
        - Extract entities and relationships
        - Visualize connections between concepts
        - Identify key topics and patterns
        """)
    
    with col3:
        st.markdown("### 🤖 AI Assistant")
        st.markdown("""
        - Ask complex questions about papers
        - Get comprehensive answers with citations
        - Connect information across multiple papers
        """)
    
    # Getting started section
    st.markdown("## Getting Started")
    
    if not st.session_state.api_keys_set:
        st.markdown('<div class="warning-box">⚠️ <b>API Key Required:</b> To use all features, please set up your Groq API key in the sidebar.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    1. Go to the **Research Papers** tab to search for papers or upload PDFs
    2. Select papers to analyze
    3. Build a knowledge graph to visualize connections
    4. Use the Query Assistant to ask questions about your papers
    """)
    
    # Technical details for interested users
    with st.expander("Technical Details"):
        st.markdown("""
        ResearchWebGraph uses a combination of cutting-edge technologies:
        
        - **Backend**: FastAPI, Qdrant vector database, SentenceTransformers
        - **Frontend**: Streamlit interactive web interface
        - **NLP**: NLTK and spaCy for entity extraction
        - **LLMs**: Groq-powered language models for question answering
        
        The system stores embeddings in a vector database for semantic search, builds knowledge graphs to represent relationships between concepts, and uses LLMs to generate natural language responses to queries about the research papers.
        """)

elif selected == "Research Papers":
    # Redirect to the Research Papers page
    from components.api import change_page
    change_page("01_Papers")

elif selected == "Knowledge Graph":
    # Redirect to the Knowledge Graph page
    from components.api import change_page
    change_page("02_Knowledge_Graph")

elif selected == "Query Assistant":
    # Redirect to the Query Assistant page
    from components.api import change_page
    change_page("03_Query_Assistant")

# Footer
st.markdown("---")
st.markdown(
    "ResearchWebGraph | Built with ❤️ using Streamlit, FastAPI, Qdrant and Groq | [GitHub](https://github.com/yourusername/ResearchWebGraph)"
)
