from fastapi import APIRouter, HTTPException, Depends, Body, Query, Path
from typing import List, Dict, Any, Optional
import logging
import os
import json
import time
from datetime import datetime
import httpx
from pydantic import BaseModel
from fastapi import Header

# Import models and schemas
from app.models.schemas import QueryRequest, QueryResponse

# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# In-memory storage for conversation history
# In a production environment, consider using a database
conversation_history = {}

class Message(BaseModel):
    """Model for a conversation message."""
    role: str
    content: str

async def get_paper_context(query: str, paper_ids: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context from papers based on the query.
    
    Args:
        query: The user's question
        paper_ids: List of paper IDs to search in
        top_k: Number of most relevant chunks to retrieve
        
    Returns:
        List of context chunks with metadata
    """
    try:
        # Import the function from process_papers service
        from app.services.process_papers import get_context_for_query
        
        # Get relevant context from the vector store
        contexts = await get_context_for_query(query, paper_ids, top_k)
        
        return contexts
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
        raise e

async def query_groq_api(messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7, groq_api_key: Optional[str] = Header(None)) -> str:
    """
    Query the Groq API with conversation messages.
    
    Args:
        messages: List of message objects with role and content
        max_tokens: Maximum tokens to generate
        temperature: Controls randomness (0 to 1)
        
    Returns:
        Generated response text
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Default to Llama 3.1 8B for quick responses, but can be adjusted
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Log the API call (without sensitive information)
        logger.info(f"Querying Groq API with model: {model}, messages count: {len(messages)}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            start_time = time.time()
            response = await client.post(url, headers=headers, json=payload)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Groq API response time: {elapsed_time:.2f}s")
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.status_code}, {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Groq API error: {response.text}"
                )
            
            response_data = response.json()
            if not response_data.get("choices") or len(response_data["choices"]) == 0:
                raise HTTPException(status_code=500, detail="No response from Groq API")
            
            return response_data["choices"][0]["message"]["content"]
            
    except httpx.TimeoutException:
        logger.error("Groq API timeout")
        raise HTTPException(status_code=504, detail="Groq API request timed out")
    except Exception as e:
        logger.error(f"Error querying Groq API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error querying Groq API: {str(e)}")

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Answer a question based on the content of selected papers.
    
    Retrieves context from the papers, uses the Groq API to generate an answer,
    and returns the response with citations to source papers.
    """
    try:
        # Validate request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not request.papers_ids:
            raise HTTPException(status_code=400, detail="At least one paper ID must be provided")
        
        # Log the query
        logger.info(f"Processing query: '{request.query}' for {len(request.papers_ids)} papers")
        
        # Get context from papers
        contexts = await get_paper_context(request.query, request.papers_ids)
        
        if not contexts:
            logger.warning(f"No relevant context found for query: '{request.query}'")
            return QueryResponse(
                answer="I couldn't find relevant information in the provided papers to answer your question.",
                sources=[]
            )
        
        # Format context for the prompt
        context_text = "\n\n".join([
            f"--- Document: {ctx['metadata']['title']} ---\n{ctx['content']}"
            for ctx in contexts
        ])
        
        # Create chat prompt with system instructions and context
        messages = [
            {
                "role": "system",
                "content": """You are a research assistant that helps analyze academic papers. 
                Your task is to provide detailed, accurate answers based on the context provided.
                When answering:
                - Be comprehensive and detailed in your explanations
                - Cite specific papers when providing information
                - Provide context for technical concepts
                - Structure your answer with clear sections where appropriate
                - If the answer cannot be determined from the context, state that clearly
                - Always maintain academic integrity and accuracy
                """
            },
            {
                "role": "user",
                "content": f"""Using the following context from research papers, please provide a detailed answer to this question:

CONTEXT:
{context_text}

QUESTION:
{request.query}
                """
            }
        ]
        
        # Generate answer using Groq API
        answer = await query_groq_api(messages)
        
        # Format sources for citation
        sources = []
        for ctx in contexts:
            metadata = ctx["metadata"]
            sources.append({
                "title": metadata["title"],
                "authors": metadata["authors"],
                "published": metadata.get("published", "Unknown"),
                "excerpt": ctx["content"][:200] + "..." if len(ctx["content"]) > 200 else ctx["content"],
                "relevance": ctx.get("relevance_score", 1.0)
            })
        
        # Log successful response
        logger.info(f"Successfully generated answer for query: '{request.query}'")
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error answering query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error answering query: {str(e)}")

@router.post("/conversation", response_model=Dict[str, Any])
async def create_conversation(
    paper_ids: List[str] = Body(..., description="List of paper IDs for the conversation")
):
    """
    Create a new conversation for interacting with papers.
    
    Initializes a new conversation context with the specified papers.
    """
    try:
        # Generate a unique conversation ID
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(conversation_history) + 1}"
        
        # Create conversation entry
        conversation_history[conversation_id] = {
            "paper_ids": paper_ids,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Created conversation {conversation_id} with {len(paper_ids)} papers")
        
        return {
            "conversation_id": conversation_id,
            "paper_ids": paper_ids,
            "message_count": 0
        }
    
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

@router.post("/conversation/{conversation_id}/message", response_model=Dict[str, Any])
async def add_message(
    conversation_id: str,
    message: str = Body(..., embed=True, description="User's message or question")
):
    """
    Add a message to an existing conversation and get a response.
    
    Adds the user's message to the conversation history and generates a response
    based on the context of the papers associated with the conversation.
    """
    try:
        # Check if conversation exists
        if conversation_id not in conversation_history:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        # Get conversation data
        conversation = conversation_history[conversation_id]
        paper_ids = conversation["paper_ids"]
        messages = conversation["messages"]
        
        # Add user message to history
        messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get context from papers
        contexts = await get_paper_context(message, paper_ids)
        
        # Format context
        context_text = ""
        if contexts:
            context_text = "\n\n".join([
                f"--- Document: {ctx['metadata']['title']} ---\n{ctx['content']}"
                for ctx in contexts
            ])
        
        # Prepare conversation for the API
        api_messages = [
            {
                "role": "system",
                "content": """You are a research assistant that helps analyze academic papers. 
                Your task is to provide detailed, accurate answers based on the context provided
                and the conversation history. Maintain continuity in the conversation.
                """
            }
        ]
        
        # Add conversation history (limited to last 5 exchanges for context)
        for msg in messages[-10:]:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add context as a separate message if available
        if context_text:
            # Insert context before the last user message
            api_messages.insert(-1, {
                "role": "user",
                "content": f"Here is additional context from the papers:\n\n{context_text}"
            })
        
        # Generate answer
        answer = await query_groq_api(api_messages)
        
        # Add assistant response to history
        messages.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update conversation metadata
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Format sources for citation
        sources = []
        if contexts:
            for ctx in contexts:
                metadata = ctx["metadata"]
                sources.append({
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "published": metadata.get("published", "Unknown"),
                    "excerpt": ctx["content"][:200] + "..." if len(ctx["content"]) > 200 else ctx["content"]
                })
        
        logger.info(f"Added message to conversation {conversation_id} and generated response")
        
        return {
            "conversation_id": conversation_id,
            "message": answer,
            "sources": sources,
            "message_count": len(messages)
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error adding message to conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding message to conversation: {str(e)}")

@router.get("/conversation/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(conversation_id: str):
    """
    Retrieve a conversation by ID.
    
    Gets the complete conversation history.
    """
    try:
        # Check if conversation exists
        if conversation_id not in conversation_history:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        # Get conversation data
        conversation = conversation_history[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "paper_ids": conversation["paper_ids"],
            "messages": conversation["messages"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "message_count": len(conversation["messages"])
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation.
    
    Removes a conversation and its history from the system.
    """
    try:
        # Check if conversation exists
        if conversation_id not in conversation_history:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        # Delete conversation
        del conversation_history[conversation_id]
        
        logger.info(f"Deleted conversation {conversation_id}")
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")
