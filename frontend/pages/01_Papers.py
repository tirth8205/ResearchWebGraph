import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

# Import components
from components.api import fetch_papers, process_pdf, get_paper
from components.sidebar import show_paper_filters

# Page configuration
st.set_page_config(
    page_title="ResearchWebGraph - Papers",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if API keys are set
if not st.session_state.get("api_keys_set", False):
    st.warning("⚠️ Please configure your API keys in the sidebar first.")

# Page header
st.markdown('<h1 class="main-header">Research Papers</h1>', unsafe_allow_html=True)

# Show paper filters in sidebar
show_paper_filters()

# Create tabs for different ways to get papers
tab1, tab2 = st.tabs(["📊 Search arXiv", "📁 Upload PDF"])

with tab1:
    st.markdown('<h2 class="sub-header">Search for Research Papers</h2>', unsafe_allow_html=True)
    
    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "Research Topic",
            placeholder="e.g., quantum computing, natural language processing",
            help="Enter a research topic to search for papers"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_papers = st.slider(
                "Maximum Papers",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of papers to retrieve"
            )
        
        with col2:
            # Show selected categories if set in filters
            categories = st.session_state.get("search_categories", None)
            date_from = st.session_state.get("search_date_from", None)
            
            if categories:
                st.info(f"Category filters: {', '.join(categories)}")
            if date_from:
                st.info(f"Date filter: From {date_from}")
        
        search_button = st.form_submit_button("🔍 Search Papers")
    
    # Handle search submission
    if search_button and query:
        # Perform search
        papers = fetch_papers(
            query=query,
            max_papers=max_papers,
            categories=st.session_state.get("search_categories"),
            date_from=st.session_state.get("search_date_from")
        )
        
        if papers:
            # Store in session state
            st.session_state.papers = papers
            st.success(f"Found {len(papers)} papers on '{query}'!")
            
            # Scroll to results - this is a small hack to improve UX
            st.markdown('<div id="search-results"></div>', unsafe_allow_html=True)
            st.markdown('<script>document.getElementById("search-results").scrollIntoView();</script>', unsafe_allow_html=True)
        else:
            st.error("No papers found. Try a different query or check your connection.")

with tab2:
    st.markdown('<h2 class="sub-header">Upload PDF Document</h2>', unsafe_allow_html=True)
    
    # PDF upload
    st.markdown("""
    Upload your own research paper or document in PDF format. 
    The system will extract text and create a paper entry that can be used for knowledge graph analysis.
    """)
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Maximum file size: 50MB")
    
    if uploaded_file is not None:
        # Display PDF details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "Upload time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Show file details
        st.json(file_details)
        
        # Process button
        if st.button("Process PDF"):
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
                st.error("File is too large. Maximum size is 50MB.")
            else:
                # Process the PDF
                paper_id = process_pdf(uploaded_file)
                
                if paper_id:
                    # Get the processed paper details
                    paper = get_paper(paper_id)
                    
                    if paper:
                        # Add to papers list
                        if "papers" not in st.session_state:
                            st.session_state.papers = []
                        
                        st.session_state.papers.append(paper)
                        
                        st.success(f"PDF processed successfully! Paper ID: {paper_id}")
                        
                        # Automatically select the paper
                        if "selected_papers" not in st.session_state:
                            st.session_state.selected_papers = []
                        
                        st.session_state.selected_papers.append(paper)
                        st.info("PDF has been added to your selected papers.")
                    else:
                        st.error("Failed to retrieve processed paper details.")
                else:
                    st.error("Failed to process PDF. Please check the file and try again.")

# Display search results
if "papers" in st.session_state and st.session_state.papers:
    st.markdown('<h2 class="sub-header">Retrieved Papers</h2>', unsafe_allow_html=True)
    
    # Actions for all results
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Select All Papers"):
            st.session_state.selected_papers = st.session_state.papers.copy()
            st.success("All papers selected!")
    
    with col2:
        if st.button("Clear Selection"):
            st.session_state.selected_papers = []
            st.info("Selection cleared.")
    
    with col3:
        # Get IDs of currently selected papers
        selected_ids = [p["id"] for p in st.session_state.get("selected_papers", [])]
        selected_count = len(selected_ids)
        st.info(f"{selected_count} papers currently selected")
    
    # Display each paper with selection option
    for i, paper in enumerate(st.session_state.papers):
        with st.expander(f"{i+1}. {paper['metadata']['title']}", expanded=i==0):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Paper metadata
                st.markdown(f"**Authors**: {', '.join(paper['metadata']['authors'])}")
                st.markdown(f"**Published**: {paper['metadata']['published']}")
                
                if paper['metadata'].get('categories'):
                    # Format categories with highlighting for CS categories
                    categories = paper['metadata']['categories']
                    formatted_categories = []
                    
                    for cat in categories:
                        if cat.startswith("cs."):
                            formatted_categories.append(f"<span style='color:#3B82F6'>{cat}</span>")
                        else:
                            formatted_categories.append(cat)
                    
                    st.markdown(f"**Categories**: {', '.join(formatted_categories)}", unsafe_allow_html=True)
                
                # Abstract
                st.markdown("### Abstract")
                st.markdown(paper['content'])
                
                # Additional metadata if available
                if paper['metadata'].get('arxiv_id'):
                    arxiv_id = paper['metadata']['arxiv_id']
                    st.markdown(f"**arXiv ID**: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
                
                if paper['metadata'].get('pdf_url'):
                    pdf_url = paper['metadata']['pdf_url']
                    st.markdown(f"**PDF**: [Download Paper]({pdf_url})")
            
            with col2:
                # Determine if paper is already selected
                paper_id = paper['id']
                is_selected = paper_id in selected_ids
                
                # Selection controls
                if is_selected:
                    if st.button("✓ Selected", key=f"selected_{paper_id}"):
                        # Remove from selected papers
                        st.session_state.selected_papers = [p for p in st.session_state.selected_papers if p['id'] != paper_id]
                        st.experimental_rerun()
                else:
                    if st.button("+ Select", key=f"select_{paper_id}"):
                        # Add to selected papers
                        if "selected_papers" not in st.session_state:
                            st.session_state.selected_papers = []
                        
                        st.session_state.selected_papers.append(paper)
                        st.experimental_rerun()
    
    # Actions after paper selection
    if st.session_state.get("selected_papers", []):
        st.markdown("---")
        st.markdown("### Next Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Build Knowledge Graph", use_container_width=True):
                from components.api import change_page
                change_page("02_Knowledge_Graph")
        
        with col2:
            if st.button("🤖 Ask Questions", use_container_width=True):
                from components.api import change_page
                change_page("03_Query_Assistant")
else:
    # Show info box if no papers are loaded yet
    st.info("Search for papers or upload a PDF to get started.")
    
    # Show a sample of arXiv categories
    with st.expander("Popular arXiv Categories", expanded=False):
        # Create a DataFrame with category information
        categories_data = {
            "Category": ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML", "physics.comp-ph", "q-bio.GN", "math.ST"],
            "Description": [
                "Artificial Intelligence",
                "Computation and Language (NLP)",
                "Computer Vision and Pattern Recognition",
                "Machine Learning",
                "Neural and Evolutionary Computing",
                "Machine Learning (Statistics)",
                "Computational Physics",
                "Genomics",
                "Statistics Theory"
            ]
        }
        
        categories_df = pd.DataFrame(categories_data)
        st.table(categories_df)

# Footer
st.markdown("---")
st.markdown(
    "Use the sidebar to manage your selected papers."
)
