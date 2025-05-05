import streamlit as st
import os
import time
import pandas as pd
from typing import List, Dict, Any
import re
from datetime import datetime

# Import components
from components.api import query_papers, create_conversation, send_message

# Page configuration
st.set_page_config(
    page_title="ResearchWebGraph - Query Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Initialize chat history in session state if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

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

# Assistant explanation
with st.expander("How to Use the Research Assistant", expanded=len(st.session_state.chat_history) == 0):
    st.markdown("""
    The Research Assistant uses AI to help you understand and explore the selected papers.
    
    ### How to Use
    
    - **Ask specific questions** about the papers' content, methodologies, findings, or relationships
    - **Get comprehensive answers** with relevant citations to the source papers
    - **Follow up with additional questions** to explore topics in more depth
    
    ### Example Questions
    
    - "What are the main findings across these papers?"
    - "Explain the methodology used in paper X"
    - "How do these papers relate to each other?"
    - "What are the limitations mentioned in these studies?"
    - "Summarize the key contributions of each paper"
    
    ### How It Works
    
    The assistant uses vector search to find relevant sections in the papers and a large language model to generate accurate, coherent answers based on the paper content.
    """)

# Create main chat interface
st.markdown('<h2 class="sub-header">Ask Questions About the Papers</h2>', unsafe_allow_html=True)

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
    max-height: 60vh;
    overflow-y: auto;
}
.user-message {
    background-color: #E9F2FF;
    padding: 10px 15px;
    border-radius: 15px 15px 0 15px;
    margin: 10px 0;
    max-width: 80%;
    align-self: flex-end;
    margin-left: auto;
}
.assistant-message {
    background-color: #F0F2F6;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    margin: 10px 0;
    max-width: 85%;
}
.source-container {
    background-color: #F8F9FA;
    border: 1px solid #E9ECEF;
    border-radius: 5px;
    padding: 8px 12px;
    margin-top: 5px;
    font-size: 0.9em;
}
.timestamp {
    font-size: 0.7em;
    color: #6c757d;
    margin-top: 5px;
    text-align: right;
}
.streaming-response {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
</style>
""", unsafe_allow_html=True)

# Function to display chat messages
def display_chat_message(role, content, timestamp=None, sources=None):
    if role == "user":
        message_class = "user-message"
        prefix = "You: "
    else:
        message_class = "assistant-message"
        prefix = "Assistant: "
    
    # Format markdown in the message content
    if role == "assistant" and isinstance(content, str):
        # Ensure code blocks are properly formatted
        content = re.sub(r'``````', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
        
        # Format bold text
        content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
        
        # Format italic text
        content = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', content)
    
    st.markdown(f'<div class="{message_class}">{content}</div>', unsafe_allow_html=True)
    
    if timestamp:
        st.markdown(f'<div class="timestamp">{timestamp}</div>', unsafe_allow_html=True)
    
    # Display sources if available
    if sources and isinstance(sources, list) and len(sources) > 0:
        with st.expander("Show sources", expanded=False):
            for i, source in enumerate(sources, 1):
                title = source.get("title", "Unknown title")
                authors = ", ".join(source.get("authors", ["Unknown"]))
                excerpt = source.get("excerpt", "")
                
                st.markdown(f"**Source {i}: {title}**")
                st.markdown(f"*Authors: {authors}*")
                st.text(excerpt)
                st.markdown("---")

# Display chat history
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.info("Ask a question about the selected papers to start a conversation.")
    else:
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            timestamp = message.get("timestamp", None)
            sources = message.get("sources", None)
            
            display_chat_message(role, content, timestamp, sources)
    
    st.markdown('</div>', unsafe_allow_html=True)

# User input
user_question = st.text_input(
    "Ask a question about the papers",
    key="user_question",
    placeholder="e.g., What are the main findings? Or, How do these papers relate to each other?",
    help="Be specific in your questions to get the most relevant answers"
)

# Clear chat button and Submit button in the same row
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation_id = None
        st.experimental_rerun()

with col2:
    submit_button = st.button("Send Question", use_container_width=True)

# Process user question when submit button is clicked
if submit_button and user_question:
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": user_question,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }
    st.session_state.chat_history.append(user_message)
    
    # Get response from assistant
    with st.spinner("Generating answer..."):
        try:
            # Get paper IDs
            paper_ids = [paper["id"] for paper in selected_papers]
            
            # Send request to backend
            if not st.session_state.conversation_id:
                # First message - create a new conversation
                conversation_id = create_conversation(paper_ids)
                if conversation_id:
                    st.session_state.conversation_id = conversation_id
                    
                # Get response using direct query for first message
                response = query_papers(user_question, paper_ids)
            else:
                # Use conversation API for follow-up messages
                response = send_message(st.session_state.conversation_id, user_question)
            
            # Process response
            if response:
                assistant_message = {
                    "role": "assistant",
                    "content": response.get("answer", response.get("message", "")),
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "sources": response.get("sources", [])
                }
                
                st.session_state.chat_history.append(assistant_message)
            else:
                error_message = {
                    "role": "assistant",
                    "content": "I encountered an error processing your question. Please try again.",
                    "timestamp": datetime.now().strftime("%I:%M %p")
                }
                st.session_state.chat_history.append(error_message)
        
        except Exception as e:
            error_message = {
                "role": "assistant",
                "content": f"Error: {str(e)}. Please try again later.",
                "timestamp": datetime.now().strftime("%I:%M %p")
            }
            st.session_state.chat_history.append(error_message)
    
    # Rerun to update UI with new messages
    st.experimental_rerun()

# Additional options
with st.expander("More Options", expanded=False):
    st.markdown("### Conversation Settings")
    
    # Model selection
    model = st.selectbox(
        "Model",
        options=[
            "llama-3.1-8b-instant (faster)",
            "llama-3.3-70b-versatile (higher quality)"
        ],
        index=0,
        help="Select which model to use for generating answers"
    )
    
    # Response length
    response_length = st.slider(
        "Max Response Length",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum number of tokens for generated responses"
    )
    
    # Save settings button
    if st.button("Save Settings"):
        # These would be passed to the backend when implemented
        model_name = model.split(" ")[0]
        st.session_state.assistant_settings = {
            "model": model_name,
            "max_tokens": response_length
        }
        st.success("Settings saved!")

# Export conversation
if st.session_state.chat_history:
    with st.expander("Export Conversation", expanded=False):
        st.markdown("Save this conversation for later reference.")
        
        # Format conversation for export
        conversation_text = ""
        for message in st.session_state.chat_history:
            role = "You" if message.get("role") == "user" else "Assistant"
            timestamp = message.get("timestamp", "")
            content = message.get("content", "")
            
            conversation_text += f"[{timestamp}] {role}:\n{content}\n\n"
        
        # Add download button
        st.download_button(
            label="Download Conversation",
            data=conversation_text,
            file_name=f"research_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown(
    "The Research Assistant uses AI to help you explore and understand the content of the selected papers. "
    "All answers are generated based on the paper content, but accuracy is not guaranteed."
)
