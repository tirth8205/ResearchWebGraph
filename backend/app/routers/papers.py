from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path, Body, File, UploadFile, Depends, status
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import asyncio
import uuid
import json
import time

from app.models.schemas import PaperRequest, PaperResponse
from app.services.fetch_papers import fetch_papers
from app.services.process_papers import process_papers, process_pdf_content
from app.utils.pdf_utils import extract_text_from_pdf, is_valid_pdf

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Track background processing tasks
active_tasks = {}

@router.post("/fetch", response_model=PaperResponse)
async def get_papers(
    request: PaperRequest,
    background_tasks: BackgroundTasks
):
    """
    Fetch research papers based on a query and process them for the knowledge graph.
    
    This endpoint searches for papers on various sources based on the provided query and optional
    filters, then processes them for use with the knowledge graph and vector store.
    """
    try:
        logger.info(f"Fetching papers with query: '{request.query}'")
        
        # Fetch papers from sources
        papers = await fetch_papers(
            query=request.query,
            max_docs=request.max_papers,
            categories=request.categories,
            date_from=request.date_from,
            sources=request.sources
        )
        
        if not papers:
            logger.warning(f"No papers found for query: '{request.query}'")
            return PaperResponse(papers=[], count=0)
        
        # Process papers in background to avoid blocking
        task_id = f"process_{datetime.now().isoformat()}"
        
        # Store task information
        active_tasks[task_id] = {
            "status": "processing",
            "papers_count": len(papers),
            "started_at": datetime.now().isoformat()
        }
        
        # Add background task for processing
        background_tasks.add_task(
            process_papers, 
            papers
        )
        
        logger.info(f"Started background task {task_id} to process {len(papers)} papers")
        
        return PaperResponse(
            papers=papers,
            count=len(papers),
            task_id=task_id
        )
    
    except Exception as e:
        logger.error(f"Error in get_papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-pdf", status_code=status.HTTP_200_OK)
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload and process a PDF file.
    
    Extracts text from the PDF and processes it for use with the knowledge graph.
    Returns a paper ID that can be used to reference the processed PDF.
    """
    try:
        logger.info(f"Processing uploaded PDF: {file.filename}")
        
        # Check file size
        max_size_mb = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {max_size_mb}MB."
            )
        
        # Validate PDF
        is_valid, error_msg = await is_valid_pdf(content)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {error_msg}")
        
        # Extract text from PDF
        text = await extract_text_from_pdf(content)
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract sufficient text from PDF. The file may be corrupted or protected."
            )
        
        # Process PDF content
        paper_id = await process_pdf_content(text, file.filename)
        
        return {"paper_id": paper_id, "message": "PDF processed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.get("/{paper_id}", status_code=status.HTTP_200_OK)
async def get_paper(paper_id: str = Path(..., description="ID of the paper to retrieve")):
    """
    Get a paper by ID.
    
    Retrieves a paper from the vector store by its unique ID.
    Returns the paper metadata and content.
    """
    try:
        logger.info(f"Retrieving paper: {paper_id}")
        
        # Import the function to avoid circular imports
        from app.services.process_papers import get_paper_by_id
        
        # Get the paper from the vector store
        paper = await get_paper_by_id(paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
        
        return paper
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving paper: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving paper: {str(e)}")

@router.get("/task/{task_id}", status_code=status.HTTP_200_OK)
async def get_task_status(task_id: str = Path(..., description="ID of the task to check")):
    """
    Get the status of a background task.
    
    Returns the current status of a paper processing task.
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return active_tasks[task_id]

@router.delete("/{paper_id}", status_code=status.HTTP_200_OK)
async def delete_paper(paper_id: str = Path(..., description="ID of the paper to delete")):
    """
    Delete a paper by ID.
    
    Removes a paper and all its associated data from the vector store.
    """
    try:
        logger.info(f"Deleting paper: {paper_id}")
        
        # Import the function to avoid circular imports
        from app.services.process_papers import delete_paper
        
        # Delete the paper from the vector store
        success = await delete_paper(paper_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found or could not be deleted")
        
        return {"message": f"Paper {paper_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting paper: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting paper: {str(e)}")
