import os
import logging
import json
import time
import streamlit as st
import requests
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def get_api_url(endpoint: str = "") -> str:
    """
    Get the full URL for an API endpoint.
    
    Args:
        endpoint: The API endpoint path (without leading slash)
        
    Returns:
        Full URL for the API endpoint
    """
    base_url = st.session_state.backend_url or os.getenv("BACKEND_URL", "http://localhost:8000")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    
    # Ensure base URL doesn't end with slash
    base_url = base_url.rstrip("/")
    
    # Ensure endpoint doesn't start with slash
    endpoint = endpoint.lstrip("/")
    
    return f"{base_url}/api/{endpoint}"

def handle_api_error(response: requests.Response, error_message: str = "API request failed") -> Dict[str, Any]:
    """
    Handle API errors with proper logging and error formatting.
    
    Args:
        response: The response object
        error_message: Base error message
        
    Returns:
        Error information dictionary
    """
    try:
        error_data = response.json()
        detail = error_data.get("detail", str(error_data))
        full_error = f"{error_message}: {detail} (Status: {response.status_code})"
    except ValueError:
        full_error = f"{error_message}: {response.text} (Status: {response.status_code})"
    
    logger.error(full_error)
    return {
        "error": True,
        "message": full_error,
        "status_code": response.status_code,
        "timestamp": datetime.now().isoformat()
    }

def fetch_papers(
    query: str,
    max_papers: int = 5,
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch papers from the backend API.
    
    Args:
        query: Search query
        max_papers: Maximum number of papers to fetch
        categories: Optional list of categories to filter by
        date_from: Optional date to filter from (YYYY-MM-DD)
        
    Returns:
        List of paper objects or empty list on error
    """
    try:
        url = get_api_url("papers/fetch")
        
        # Prepare request data
        data = {
            "query": query,
            "max_papers": max_papers,
            "categories": categories,
            "date_from": date_from
        }
        
        # Show spinner during API call
        with st.spinner("Fetching research papers..."):
            # Make request
            response = requests.post(
                url, 
                json=data,
                timeout=60  # Longer timeout for paper fetching
            )
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            papers = result.get("papers", [])
            logger.info(f"Successfully fetched {len(papers)} papers")
            return papers
        else:
            error_info = handle_api_error(response, "Error fetching papers")
            st.error(error_info["message"])
            return []
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while fetching papers")
        st.error("Request timed out. The server might be busy or unavailable.")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while fetching papers")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return []
    except Exception as e:
        logger.error(f"Exception fetching papers: {str(e)}", exc_info=True)
        st.error(f"Error fetching papers: {str(e)}")
        return []

def process_pdf(file) -> Optional[str]:
    """
    Upload and process a PDF file.
    
    Args:
        file: PDF file object from file_uploader
        
    Returns:
        Paper ID if successful, None otherwise
    """
    try:
        url = get_api_url("papers/upload-pdf")
        
        # Prepare file for upload
        files = {"file": (file.name, file, "application/pdf")}
        
        # Show spinner during upload
        with st.spinner("Uploading and processing PDF..."):
            # Make request
            response = requests.post(url, files=files, timeout=60)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            paper_id = result.get("paper_id")
            logger.info(f"Successfully processed PDF: {file.name}, ID: {paper_id}")
            return paper_id
        else:
            error_info = handle_api_error(response, "Error processing PDF")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while processing PDF")
        st.error("Request timed out. The server might be busy or the PDF is too large.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while processing PDF")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception processing PDF: {str(e)}", exc_info=True)
        st.error(f"Error processing PDF: {str(e)}")
        return None

def build_graph(paper_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    Build a knowledge graph from selected papers.
    
    Args:
        paper_ids: List of paper IDs
        
    Returns:
        Graph data if successful, None otherwise
    """
    try:
        url = get_api_url("graph/build")
        
        # Show spinner during graph building
        with st.spinner("Building knowledge graph..."):
            # Make request
            response = requests.post(url, json=paper_ids, timeout=120)
        
        # Check for successful response
        if response.status_code == 200:
            graph_data = response.json()
            logger.info(f"Successfully built graph with {len(graph_data.get('nodes', []))} nodes")
            
            # Store graph ID in session state
            if "id" in graph_data:
                st.session_state.current_graph_id = graph_data["id"]
            
            return graph_data
        else:
            error_info = handle_api_error(response, "Error building graph")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while building graph")
        st.error("Request timed out. Building the knowledge graph is taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while building graph")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception building graph: {str(e)}", exc_info=True)
        st.error(f"Error building graph: {str(e)}")
        return None

def get_visualization(graph_id: str, title: str = "Research Knowledge Graph") -> Optional[str]:
    """
    Get HTML visualization for a knowledge graph.
    
    Args:
        graph_id: ID of the graph to visualize
        title: Title for the visualization
        
    Returns:
        HTML content if successful, None otherwise
    """
    try:
        url = get_api_url("graph/visualize")
        
        # Prepare request data
        data = {
            "graph_id": graph_id,
            "title": title,
            "height": "700px",
            "width": "100%"
        }
        
        # Show spinner during visualization
        with st.spinner("Generating visualization..."):
            # Make request
            response = requests.post(url, json=data, timeout=60)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            html_content = result.get("html_content")
            logger.info(f"Successfully retrieved visualization for graph {graph_id}")
            return html_content
        else:
            error_info = handle_api_error(response, "Error getting visualization")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while getting visualization")
        st.error("Request timed out. Generating the visualization is taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while getting visualization")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception getting visualization: {str(e)}", exc_info=True)
        st.error(f"Error getting visualization: {str(e)}")
        return None

def query_papers(question: str, paper_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    Ask a question about the papers.
    
    Args:
        question: User's question
        paper_ids: List of paper IDs to query
        
    Returns:
        Response data if successful, None otherwise
    """
    try:
        url = get_api_url("query/ask")
        
        # Prepare request data
        data = {
            "query": question,
            "papers_ids": paper_ids
        }
        
        # Show spinner during query
        with st.spinner("Generating answer..."):
            # Make request
            response = requests.post(url, json=data, timeout=120)  # Longer timeout for LLM processing
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully queried papers with question: '{question}'")
            return result
        else:
            error_info = handle_api_error(response, "Error querying papers")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while querying papers")
        st.error("Request timed out. Generating the answer is taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while querying papers")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception querying papers: {str(e)}", exc_info=True)
        st.error(f"Error querying papers: {str(e)}")
        return None

def create_conversation(paper_ids: List[str]) -> Optional[str]:
    """
    Create a new conversation for the selected papers.
    
    Args:
        paper_ids: List of paper IDs for the conversation
        
    Returns:
        Conversation ID if successful, None otherwise
    """
    try:
        url = get_api_url("query/conversation")
        
        # Make request
        response = requests.post(url, json=paper_ids, timeout=30)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            conversation_id = result.get("conversation_id")
            logger.info(f"Successfully created conversation {conversation_id}")
            return conversation_id
        else:
            error_info = handle_api_error(response, "Error creating conversation")
            st.error(error_info["message"])
            return None
    
    except Exception as e:
        logger.error(f"Exception creating conversation: {str(e)}", exc_info=True)
        st.error(f"Error creating conversation: {str(e)}")
        return None

def send_message(conversation_id: str, message: str) -> Optional[Dict[str, Any]]:
    """
    Send a message to a conversation.
    
    Args:
        conversation_id: ID of the conversation
        message: User's message
        
    Returns:
        Response data if successful, None otherwise
    """
    try:
        url = get_api_url(f"query/conversation/{conversation_id}/message")
        
        # Prepare request data
        data = {"message": message}
        
        # Show spinner during processing
        with st.spinner("Generating response..."):
            # Make request
            response = requests.post(url, json=data, timeout=120)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully sent message to conversation {conversation_id}")
            return result
        else:
            error_info = handle_api_error(response, "Error sending message")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while sending message")
        st.error("Request timed out. Generating the response is taking too long.")
        return None
    except Exception as e:
        logger.error(f"Exception sending message: {str(e)}", exc_info=True)
        st.error(f"Error sending message: {str(e)}")
        return None

def get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific paper by ID.
    
    Args:
        paper_id: ID of the paper to retrieve
        
    Returns:
        Paper data if successful, None otherwise
    """
    try:
        url = get_api_url(f"papers/{paper_id}")
        
        # Make request
        response = requests.get(url, timeout=30)
        
        # Check for successful response
        if response.status_code == 200:
            paper = response.json()
            logger.info(f"Successfully retrieved paper {paper_id}")
            return paper
        else:
            error_info = handle_api_error(response, "Error retrieving paper")
            st.error(error_info["message"])
            return None
    
    except Exception as e:
        logger.error(f"Exception retrieving paper: {str(e)}", exc_info=True)
        st.error(f"Error retrieving paper: {str(e)}")
        return None

def change_page(page_name: str) -> None:
    """
    Change to a different page in the Streamlit multi-page app.
    
    Args:
        page_name: Name of the page to navigate to
    """import os
import logging
import json
import time
import streamlit as st
import requests
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def get_api_url(endpoint: str = "") -> str:
    """
    Get the full URL for an API endpoint.
    
    Args:
        endpoint: The API endpoint path (without leading slash)
        
    Returns:
        Full URL for the API endpoint
    """
    base_url = st.session_state.backend_url or os.getenv("BACKEND_URL", "http://localhost:8000")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    
    # Ensure base URL doesn't end with slash
    base_url = base_url.rstrip("/")
    
    # Ensure endpoint doesn't start with slash
    endpoint = endpoint.lstrip("/")
    
    return f"{base_url}/api/{endpoint}"

def handle_api_error(response: requests.Response, error_message: str = "API request failed") -> Dict[str, Any]:
    """
    Handle API errors with proper logging and error formatting.
    
    Args:
        response: The response object
        error_message: Base error message
        
    Returns:
        Error information dictionary
    """
    try:
        error_data = response.json()
        detail = error_data.get("detail", str(error_data))
        full_error = f"{error_message}: {detail} (Status: {response.status_code})"
    except ValueError:
        full_error = f"{error_message}: {response.text} (Status: {response.status_code})"
    
    logger.error(full_error)
    return {
        "error": True,
        "message": full_error,
        "status_code": response.status_code,
        "timestamp": datetime.now().isoformat()
    }

def fetch_papers(
    query: str,
    max_papers: int = 5,
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    sources: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch papers from the backend API.
    
    Args:
        query: Search query
        max_papers: Maximum number of papers to fetch per source
        categories: Optional list of categories to filter by
        date_from: Optional date to filter from (YYYY-MM-DD)
        sources: Optional list of sources to query
        
    Returns:
        List of paper objects or empty list on error
    """
    try:
        url = get_api_url("papers/fetch")
        
        # Prepare request data
        data = {
            "query": query,
            "max_papers": max_papers,
            "categories": categories,
            "date_from": date_from,
            "sources": sources
        }
        
        # Show spinner during API call
        with st.spinner("Fetching research papers..."):
            # Make request
            response = requests.post(
                url, 
                json=data,
                timeout=60  # Longer timeout for paper fetching
            )
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            papers = result.get("papers", [])
            logger.info(f"Successfully fetched {len(papers)} papers")
            return papers
        else:
            error_info = handle_api_error(response, "Error fetching papers")
            st.error(error_info["message"])
            return []
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while fetching papers")
        st.error("Request timed out. The server might be busy or unavailable.")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while fetching papers")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return []
    except Exception as e:
        logger.error(f"Exception fetching papers: {str(e)}", exc_info=True)
        st.error(f"Error fetching papers: {str(e)}")
        return []

def process_pdf(file) -> Optional[str]:
    """
    Upload and process a PDF file.
    
    Args:
        file: PDF file object from file_uploader
        
    Returns:
        Paper ID if successful, None otherwise
    """
    try:
        url = get_api_url("papers/upload-pdf")
        
        # Prepare file for upload
        files = {"file": (file.name, file, "application/pdf")}
        
        # Show spinner during upload
        with st.spinner("Uploading and processing PDF..."):
            # Make request
            response = requests.post(url, files=files, timeout=60)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            paper_id = result.get("paper_id")
            logger.info(f"Successfully processed PDF: {file.name}, ID: {paper_id}")
            return paper_id
        else:
            error_info = handle_api_error(response, "Error processing PDF")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while processing PDF")
        st.error("Request timed out. The server might be busy or the PDF is too large.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while processing PDF")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception processing PDF: {str(e)}", exc_info=True)
        st.error(f"Error processing PDF: {str(e)}")
        return None

def build_graph(paper_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    Build a knowledge graph from selected papers.
    
    Args:
        paper_ids: List of paper IDs
        
    Returns:
        Graph data if successful, None otherwise
    """
    try:
        url = get_api_url("graph/build")
        
        # Show spinner during graph building
        with st.spinner("Building knowledge graph..."):
            # Make request
            response = requests.post(url, json=paper_ids, timeout=120)
        
        # Check for successful response
        if response.status_code == 200:
            graph_data = response.json()
            logger.info(f"Successfully built graph with {len(graph_data.get('nodes', []))} nodes")
            
            # Store graph ID in session state
            if "id" in graph_data:
                st.session_state.current_graph_id = graph_data["id"]
            
            return graph_data
        else:
            error_info = handle_api_error(response, "Error building graph")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while building graph")
        st.error("Request timed out. Building the knowledge graph is taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while building graph")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception building graph: {str(e)}", exc_info=True)
        st.error(f"Error building graph: {str(e)}")
        return None

def get_visualization(graph_id: str, title: str = "Research Knowledge Graph") -> Optional[str]:
    """
    Get HTML visualization for a knowledge graph.
    
    Args:
        graph_id: ID of the graph to visualize
        title: Title for the visualization
        
    Returns:
        HTML content if successful, None otherwise
    """
    try:
        url = get_api_url("graph/visualize")
        
        # Prepare request data
        data = {
            "graph_id": graph_id,
            "title": title,
            "height": "700px",
            "width": "100%"
        }
        
        # Show spinner during visualization
        with st.spinner("Generating visualization..."):
            # Make request
            response = requests.post(url, json=data, timeout=60)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            html_content = result.get("html_content")
            logger.info(f"Successfully retrieved visualization for graph {graph_id}")
            return html_content
        else:
            error_info = handle_api_error(response, "Error getting visualization")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while getting visualization")
        st.error("Request timed out. Generating the visualization is taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while getting visualization")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception getting visualization: {str(e)}", exc_info=True)
        st.error(f"Error getting visualization: {str(e)}")
        return None

def query_papers(question: str, paper_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    Ask a question about the papers.
    
    Args:
        question: User's question
        paper_ids: List of paper IDs to query
        
    Returns:
        Response data if successful, None otherwise
    """
    try:
        url = get_api_url("query/ask")
        
        # Prepare request data
        data = {
            "query": question,
            "papers_ids": paper_ids
        }
        
        # Show spinner during query
        with st.spinner("Generating answer..."):
            # Make request
            response = requests.post(url, json=data, timeout=120)  # Longer timeout for LLM processing
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully queried papers with question: '{question}'")
            return result
        else:
            error_info = handle_api_error(response, "Error querying papers")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while querying papers")
        st.error("Request timed out. Generating the answer is taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while querying papers")
        st.error("Cannot connect to the server. Please check if the backend is running.")
        return None
    except Exception as e:
        logger.error(f"Exception querying papers: {str(e)}", exc_info=True)
        st.error(f"Error querying papers: {str(e)}")
        return None

def create_conversation(paper_ids: List[str]) -> Optional[str]:
    """
    Create a new conversation for the selected papers.
    
    Args:
        paper_ids: List of paper IDs for the conversation
        
    Returns:
        Conversation ID if successful, None otherwise
    """
    try:
        url = get_api_url("query/conversation")
        
        # Make request
        response = requests.post(url, json=paper_ids, timeout=30)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            conversation_id = result.get("conversation_id")
            logger.info(f"Successfully created conversation {conversation_id}")
            return conversation_id
        else:
            error_info = handle_api_error(response, "Error creating conversation")
            st.error(error_info["message"])
            return None
    
    except Exception as e:
        logger.error(f"Exception creating conversation: {str(e)}", exc_info=True)
        st.error(f"Error creating conversation: {str(e)}")
        return None

def send_message(conversation_id: str, message: str) -> Optional[Dict[str, Any]]:
    """
    Send a message to a conversation.
    
    Args:
        conversation_id: ID of the conversation
        message: User's message
        
    Returns:
        Response data if successful, None otherwise
    """
    try:
        url = get_api_url(f"query/conversation/{conversation_id}/message")
        
        # Prepare request data
        data = {"message": message}
        
        # Show spinner during processing
        with st.spinner("Generating response..."):
            # Make request
            response = requests.post(url, json=data, timeout=120)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully sent message to conversation {conversation_id}")
            return result
        else:
            error_info = handle_api_error(response, "Error sending message")
            st.error(error_info["message"])
            return None
    
    except requests.exceptions.Timeout:
        logger.error("Request timed out while sending message")
        st.error("Request timed out. Generating the response is taking too long.")
        return None
    except Exception as e:
        logger.error(f"Exception sending message: {str(e)}", exc_info=True)
        st.error(f"Error sending message: {str(e)}")
        return None

def get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific paper by ID.
    
    Args:
        paper_id: ID of the paper to retrieve
        
    Returns:
        Paper data if successful, None otherwise
    """
    try:
        url = get_api_url(f"papers/{paper_id}")
        
        # Make request
        response = requests.get(url, timeout=30)
        
        # Check for successful response
        if response.status_code == 200:
            paper = response.json()
            logger.info(f"Successfully retrieved paper {paper_id}")
            return paper
        else:
            error_info = handle_api_error(response, "Error retrieving paper")
            st.error(error_info["message"])
            return None
    
    except Exception as e:
        logger.error(f"Exception retrieving paper: {str(e)}", exc_info=True)
        st.error(f"Error retrieving paper: {str(e)}")
        return None

def change_page(page_name: str) -> None:
    """
    Change to a different page in the Streamlit multi-page app.
    
    Args:
        page_name: Name of the page to navigate to
    """
    # Get the base URL of the current page
    if not page_name.startswith("pages/"):
        page_name = f"pages/{page_name}"
    
    # Use the newer st.query_params to set the page parameter
    st.query_params["page"] = page_name
    time.sleep(0.1)  # Small delay to allow query params to propagate
    st.rerun()  # Use st.rerun() instead of experimental_rerun()

    # Get the base URL of the current page
    if not page_name.startswith("pages/"):
        page_name = f"pages/{page_name}"
    
    # Use the newer st.query_params instead of experimental_set_query_params
    st.query_params["page"] = page_name
    time.sleep(0.1)  # Small delay to allow query params to propagate
    st.rerun()  # Use st.rerun() instead of experimental_rerun()
