from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
import time
from contextlib import asynccontextmanager

# Import routers
from app.routers import papers, knowledge_graph, query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Load or initialize any resources needed
    logger.info("Starting ResearchWebGraph API...")
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set. Some features may not work correctly.")
    
    # Initialize any connections or resources
    # This is where you would initialize Qdrant, SentenceTransformers, etc.
    
    yield  # Application runs here
    
    # Shutdown: Clean up any resources
    logger.info("Shutting down ResearchWebGraph API...")

# Initialize FastAPI app with metadata and lifespan
app = FastAPI(
    title="ResearchWebGraph API",
    description="API for fetching research papers, building knowledge graphs, and answering queries",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track and log request processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        logger.warning(f"Slow request: {request.method} {request.url.path} - {process_time:.3f}s")
    
    return response

# Include API routers
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(knowledge_graph.router, prefix="/api/graph", tags=["knowledge graph"])
app.include_router(query.router, prefix="/api/query", tags=["query"])

@app.get("/")
async def root():
    """Root endpoint that shows API information."""
    return {
        "message": "ResearchWebGraph API is running",
        "documentation": "/docs",
        "version": app.version,
        "features": [
            "Research paper retrieval from arXiv",
            "PDF document processing",
            "Knowledge graph generation and visualization",
            "Question answering with Groq LLM"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    # Check any critical services here
    # For instance, check if Qdrant is accessible
    
    return {
        "status": "healthy",
        "api_version": app.version,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY")),
    }

# Run the application with Uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Start Uvicorn server
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )
