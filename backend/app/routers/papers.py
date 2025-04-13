from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Query
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime

# Import models and services
from app.models.schemas import PaperRequest, PaperResponse, Paper, PDFUploadResponse
from app.services.fetch_papers import fetch_papers
from app.services.process_papers import process_papers, process_pdf_content

# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Background task tracking
active_tasks = {}

@router.post("/fetch", response_model=PaperResponse)
async def get_papers(
    request: PaperRequest,
    background_tasks: BackgroundTasks
):
    """
    Fetch research papers based on a query and process them for the knowledge graph.
    
    This endpoint searches for papers on arXiv based on the provided query and optional
    filters, then processes them for use with the knowledge graph and vector store.
    """
    try:
        logger.info(f"Fetching papers with query: '{request.query}'")
        
        # Fetch papers from arXiv
        papers = await fetch_papers(
            query=request.query,
            max_docs=request.max_papers,
            categories=request.categories,
            date_from=request.date_from
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
        
        # Process papers asynchronously
        processed_papers = await process_papers(papers)
        
        # Update task status
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Successfully fetched and processed {len(processed_papers)} papers")
        
        return PaperResponse(
            papers=processed_papers,
            count=len(processed_papers)
        )
    
    except Exception as e:
        logger.error(f"Error fetching papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {str(e)}")

@router.get("/status/{task_id}")
async def get_processing_status(task_id: str):
    """
    Check the status of a paper processing job.
    
    Returns the current status of a background processing task.
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return active_tasks[task_id]

@router.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a PDF file to extract text and add to the knowledge base.
    
    Processes a PDF document, extracts its text, and adds it to the vector store
    for later querying and knowledge graph generation.
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        logger.info(f"Processing PDF: {file.filename}")
        
        # Read the file content
        file_content = await file.read()
        
        # Extract text from PDF
        from app.utils.pdf_utils import extract_text_from_pdf
        text = await extract_text_from_pdf(file_content)
        
        if not text or len(text.strip()) < 100:  # Arbitrary minimum length
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        # Process the PDF content
        paper_id = await process_pdf_content(text, file.filename)
        
        logger.info(f"Successfully processed PDF: {file.filename}, assigned ID: {paper_id}")
        
        return PDFUploadResponse(
            paper_id=paper_id,
            filename=file.filename,
            success=True,
            message="PDF processed successfully"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.get("/list", response_model=PaperResponse)
async def list_papers(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    category: Optional[str] = None
):
    """
    List processed papers with optional filtering.
    
    Retrieves papers that have been processed and stored in the system.
    """
    try:
        # Implement paper retrieval from your storage (Qdrant)
        # This is a placeholder implementation
        from app.services.process_papers import list_stored_papers
        
        papers = await list_stored_papers(limit, offset, category)
        
        return PaperResponse(
            papers=papers,
            count=len(papers)
        )
    
    except Exception as e:
        logger.error(f"Error listing papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing papers: {str(e)}")

@router.get("/{paper_id}", response_model=Paper)
async def get_paper(paper_id: str):
    """
    Retrieve a specific paper by ID.
    
    Gets the complete information for a single paper.
    """
    try:
        # Implement retrieval of a single paper
        from app.services.process_papers import get_paper_by_id
        
        paper = await get_paper_by_id(paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
        
        return paper
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving paper: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving paper: {str(e)}")

@router.delete("/{paper_id}")
async def delete_paper(paper_id: str):
    """
    Delete a paper from the system.
    
    Removes a paper and its associated data from the vector store and knowledge graph.
    """
    try:
        # Implement paper deletion
        from app.services.process_papers import delete_paper
        
        success = await delete_paper(paper_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
        
        return {"message": f"Paper {paper_id} deleted successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error deleting paper: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting paper: {str(e)}")
