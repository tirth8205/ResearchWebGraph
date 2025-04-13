import streamlit as st
import os
import time
import pandas as pd
from typing import List, Dict, Any
import random
from datetime import datetime

# Import components
from components.api import query_papers, create_conversation, send_message
from components.sidebar import show_graph_settings

# Page configuration
st.set_page_config(
    page_title="ResearchWebGraph - Query Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #E9F2FF;
        border-left: 5px solid #2E86C1;
    }
    .chat-message.assistant {
        background-color: #F0F0F0;
        border-left: 5px solid #5D6D7E;
    }
    .chat-message .message-content {
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #F9F9F9;
        border: 1px solid #E0E0E0;
        border-radius: 0.3rem;
        padding: 0.7rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .source-title {
        font-weight: bold;
        color: #1A5276;
    }
    .source-authors {
        font-style: italic;
        color: #566573;
    }
    .source-excerpt {
        margin-top: 0.3rem;
        color: #424949;
    }
    .thinking-animation {
        display: inline-block;
        margin-left: 10px;
    }
    .thinking-animation span {
        animation: thinking 1.4s infinite;
        display: inline-block;
        margin-right: 2px;
    }
    .thinking-animation span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .thinking-animation span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes thinking {
        0%, 80%, 100% { transform: scale(0); opacity: 0; }
        40% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Check if API keys are set
if not st.session_state.get("api_keys_set", False):
    st.warning("⚠️ Please configure your API keys in the sidebar first.")
    st.stop()

# Page header
st.markdown('<h1 class="main-header">Research Assistant</h1>', unsafe_allow_html=True)

# First, check if papers are selected
selected_papers = st.session_state.get("selected_papers", [])

if not selected_papers:
    st.warning("⚠️ No papers selected. Please select papers in the Research Papers tab first.")
    
    # Add a navigation button
    if st.button("Go to Paper Selection"):
        from components.api import change_page
        change_page("01_Papers")
    
    st.stop()

# Initialize chat history if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize conversation ID if not exists
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Display selected papers summary
with st.expander("Selected Papers", expanded=False):
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

# Main chat interface
st.markdown('<h2 class="sub-header">Ask Questions About the Papers</h2>', unsafe_allow_html=True)

# Instructions for the user
with st.expander("How to Use the Research Assistant", expanded=not bool(st.session_state.chat_history)):
    st.markdown("""
    The Research Assistant helps you understand and explore the papers you've selected. Here's what you can ask:

    ### Example Questions
    
    - **Summarization**: "Summarize the key findings from these papers."
    - **Comparisons**: "What are the main differences in approach between these papers?"
    - **Methods**: "Explain the methodology used in these studies."
    - **Applications**: "What are the practical applications mentioned in these papers?"
    - **Technical Details**: "Explain the mathematical model described in the second paper."
    - **Connections**: "How do these papers relate to each other?"
    
    ### Tips for Better Results
    
    - Be specific in your questions
    - Ask one question at a time for more focused answers
    - For complex topics, break down into multiple questions
    - The assistant will cite sources from the papers to support its answers
    """)

# Create two columns - one for chat history and one for information
chat_col, info_col = st.columns([3, 1])

# Chat history column
with chat_col:
    # Create a container for the chat history
    chat_container = st.container(height=500)
    
    # Display chat history
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Ask a question to start the conversation.")
        else:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f'<div class="chat-message user"><div class="message-content">👤 <b>You:</b> {message}</div></div>', unsafe_allow_html=True)
                else:
                    # Format assistant message with sources if available
                    if isinstance(message, dict):
                        answer = message.get("answer", "")
                        sources = message.get("sources", [])
                        
                        # Main answer
                        message_html = f'<div class="chat-message assistant"><div class="message-content">🤖 <b>Assistant:</b> {answer}</div>'
                        
                        # Add sources if available
                        if sources:
                            message_html += '<div class="message-sources"><p><b>Sources:</b></p>'
                            for source in sources:
                                title = source.get("title", "Untitled")
                                authors = ", ".join(source.get("authors", ["Unknown"]))
                                excerpt = source.get("excerpt", "")
                                
                                message_html += f'''
                                <div class="source-box">
                                    <div class="source-title">{title}</div>
                                    <div class="source-authors">{authors}</div>
                                    <div class="source-excerpt">{excerpt}</div>
                                </div>
                                '''
                            
                            message_html += '</div>'
                        
                        message_html += '</div>'
                        st.markdown(message_html, unsafe_allow_html=True)
                    else:
                        # Simple text response
                        st.markdown(f'<div class="chat-message assistant"><div class="message-content">🤖 <b>Assistant:</b> {message}</div></div>', unsafe_allow_html=True)

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            key="user_input",
            help="Ask any question about the selected papers",
            placeholder="e.g., Summarize the key findings from these papers.",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button(
                label="Ask", 
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            clear_button = st.form_submit_button(
                label="Clear Chat", 
                use_container_width=True
            )
    
    if submit_button and user_question:
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_question))
        
        # Display "thinking" message
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '''<div class="chat-message assistant">
                <div class="message-content">🤖 <b>Assistant:</b> Thinking
                    <div class="thinking-animation">
                        <span>.</span><span>.</span><span>.</span>
                    </div>
                </div>
            </div>''', 
            unsafe_allow_html=True
        )
        
        # Get paper IDs
        paper_ids = [paper["id"] for paper in selected_papers]
        
        try:
            # If this is a new conversation, create one
            if not st.session_state.conversation_id:
                # Choose between different querying approaches
                # 1. Simple one-shot query
                response = query_papers(user_question, paper_ids)
                
                if response:
                    # Add assistant response to chat history
                    st.session_state.chat_history.append(("assistant", response))
                else:
                    st.session_state.chat_history.append(("assistant", "I encountered an error processing your question. Please try again."))
            
            else:
                # Use existing conversation
                response = send_message(st.session_state.conversation_id, user_question)
                
                if response:
                    message = response.get("message", "")
                    sources = response.get("sources", [])
                    
                    # Format response with sources
                    response_with_sources = {
                        "answer": message,
                        "sources": sources
                    }
                    
                    # Add to chat history
                    st.session_state.chat_history.append(("assistant", response_with_sources))
                else:
                    st.session_state.chat_history.append(("assistant", "I encountered an error processing your question. Please try again."))
        
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"Error: {str(e)}"))
        
        # Remove thinking message
        thinking_placeholder.empty()
        
        # Rerun to update the UI
        st.experimental_rerun()
    
    elif clear_button:
        # Clear chat history
        st.session_state.chat_history = []
        st.session_state.conversation_id = None
        
        # Rerun to update the UI
        st.experimental_rerun()

# Information column
with info_col:
    st.markdown("### Research Context")
    
    # Count of papers
    st.metric("Papers Analyzed", len(selected_papers))
    
    # Paper word count (approximate)
    total_words = sum(len(paper.get("content", "").split()) for paper in selected_papers)
    st.metric("Total Word Count", f"{total_words:,}")
    
    # Calculate publication range
    publication_dates = []
    for paper in selected_papers:
        published = paper.get("metadata", {}).get("published", "")
        if published:
            try:
                # Try to extract just the date part if it's a datetime
                if "T" in published:
                    published = published.split("T")[0]
                publication_dates.append(published)
            except:
                pass
    
    if publication_dates:
        st.metric("Publication Range", f"{min(publication_dates)} - {max(publication_dates)}")
    
    # Topic suggestions
    st.markdown("### Suggested Questions")
    
    suggestion_categories = {
        "Summary": [
            "Summarize the key findings from these papers.",
            "What are the main contributions of these papers?",
            "What problems do these papers address?"
        ],
        "Methodology": [
            "What methods are used in these papers?",
            "How do the experimental setups differ?",
            "What datasets were used in these studies?"
        ],
        "Comparison": [
            "How do the approaches in these papers compare?",
            "What are the strengths and weaknesses of each paper?",
            "How do these papers relate to each other?"
        ],
        "Applications": [
            "What are the practical applications mentioned?",
            "How could these findings be implemented?",
            "What industries could benefit from this research?"
        ]
    }
    
    # Create tabs for suggestion categories
    suggestion_tabs = st.tabs(list(suggestion_categories.keys()))
    
    for i, (category, suggestions) in enumerate(suggestion_categories.items()):
        with suggestion_tabs[i]:
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggestion_{category}_{suggestion}", use_container_width=True):
                    # Set the suggestion as the user input (we'll need to rerun for this)
                    st.session_state.user_input = suggestion
                    # Submit the form programmatically (doesn't work directly)
                    # Instead, we'll add to chat history and rerun
                    st.session_state.chat_history.append(("user", suggestion))
                    # Rerun to update UI and trigger query on next run
                    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    "The Research Assistant uses semantic search and LLMs to answer questions based specifically on the content of your selected papers."
)
