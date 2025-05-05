import logging
import os
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

# Import vector database and embedding model
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range
)
from sentence_transformers import SentenceTransformer

# Import utils for text processing
from app.utils.nlp_utils import split_text_into_chunks

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "research_papers")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Lazily initialized clients and models
_async_qdrant_client = None
_sync_qdrant_client = None
_embedding_model = None

async def get_async_qdrant_client() -> AsyncQdrantClient:
    """Get or initialize the async Qdrant client."""
    global _async_qdrant_client
    if _async_qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        
        logger.info(f"Initializing async Qdrant client with URL: {url}")
        _async_qdrant_client = AsyncQdrantClient(
            url=url,
            api_key=api_key if api_key else None
        )
    
    return _async_qdrant_client

def get_sync_qdrant_client() -> QdrantClient:
    """Get or initialize the synchronous Qdrant client."""
    global _sync_qdrant_client
    if _sync_qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        
        logger.info(f"Initializing sync Qdrant client with URL: {url}")
        _sync_qdrant_client = QdrantClient(
            url=url,
            api_key=api_key if api_key else None
        )
    
    return _sync_qdrant_client

def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        model_name = DEFAULT_EMBEDDING_MODEL
        
        logger.info(f"Loading SentenceTransformer embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    
    return _embedding_model

async def setup_collection(collection_name: str = DEFAULT_COLLECTION_NAME) -> str:
    """
    Create or get the Qdrant collection for paper embeddings.
    
    Args:
        collection_name: Name of the collection to create or get
        
    Returns:
        Name of the collection
    """
    try:
        client = await get_async_qdrant_client()
        
        # Check if collection exists
        collections = await client.get_collections()
        if collection_name not in [c.name for c in collections.collections]:
            # Get embedding dimension from model
            model = get_embedding_model()
            embedding_dim = model.get_sentence_embedding_dimension()
            
            # Create the collection with the right vector dimensions
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            # Create necessary payload indexes for efficient filtering
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="paper_id",
                field_schema="keyword"
            )
            
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="chunk_index",
                field_schema="integer"
            )
            
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.categories",
                field_schema="keyword"
            )
            
            logger.info(f"Created new collection: {collection_name} with dimension {embedding_dim}")
        else:
            logger.info(f"Using existing collection: {collection_name}")
        
        return collection_name
        
    except Exception as e:
        logger.error(f"Error setting up Qdrant collection: {str(e)}", exc_info=True)
        raise

async def process_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process papers for vector storage and retrieval.
    
    Args:
        papers: List of paper dictionaries with id, content, and metadata
        
    Returns:
        The processed papers with any additional processing metadata
    """
    if not papers:
        logger.warning("No papers provided for processing")
        return []
    
    try:
        # Setup Qdrant collection
        collection_name = await setup_collection()
        
        # Get clients and model
        client = await get_async_qdrant_client()
        model = get_embedding_model()
        
        # Process each paper
        processed_papers = []
        for paper in papers:
            try:
                paper_id = paper["id"]
                content = paper["content"]
                metadata = paper["metadata"]
                
                logger.info(f"Processing paper {paper_id}: {metadata.get('title', 'Untitled')}")
                
                # Split content into chunks
                chunks = split_text_into_chunks(
                    content,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP
                )
                logger.info(f"Split paper {paper_id} into {len(chunks)} chunks")
                
                # Process chunks in batches for efficiency
                chunk_batch_size = 10  # How many chunks to process at once
                for i in range(0, len(chunks), chunk_batch_size):
                    batch = chunks[i:i+chunk_batch_size]
                    
                    # Generate embeddings for the batch
                    embeddings = model.encode(batch)
                    
                    # Prepare points for Qdrant
                    points = []
                    for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                        chunk_index = i + j
                        point_id = str(uuid.uuid4())  # Generate UUID for each chunk
                        
                        points.append(PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload={
                                "paper_id": paper_id,
                                "chunk_index": chunk_index,
                                "content": chunk,
                                "metadata": metadata,
                                "created_at": datetime.now().isoformat()
                            }
                        ))
                    
                    # Upsert the batch to Qdrant
                    await client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                
                # Add to processed papers
                processed_papers.append(paper)
                logger.info(f"Successfully processed paper {paper_id}")
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('id', 'unknown')}: {str(e)}", exc_info=True)
                # Continue with other papers
        logger.info(f"Processed {len(processed_papers)} papers")
        return processed_papers
        
    except Exception as e:
        logger.error(f"Error in paper processing pipeline: {str(e)}", exc_info=True)
        raise

async def process_pdf_content(text: str, filename: str) -> str:
    """
    Process text extracted from a PDF.
    
    Args:
        text: Extracted text content from the PDF
        filename: Name of the original PDF file
        
    Returns:
        ID of the processed paper
    """
    # Generate a unique paper ID
    paper_id = f"pdf_{uuid.uuid4()}"
    
    # Create a paper object
    paper = {
        "id": paper_id,
        "content": text,
        "metadata": {
            "title": filename,
            "authors": ["Unknown"],  # PDF extraction doesn't know authors
            "published": "Unknown",
            "pdf_url": None,
            "source": "pdf_upload"
        }
    }
    
    # Process the paper
    await process_papers([paper])
    
    return paper_id

async def get_context_for_query(query: str, paper_ids: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context from the vector store based on a query.
    
    Args:
        query: The user's question
        paper_ids: List of paper IDs to search within
        top_k: Number of top results to return
        
    Returns:
        List of context chunks with metadata and relevance scores
    """
    try:
        collection_name = DEFAULT_COLLECTION_NAME
        client = await get_async_qdrant_client()
        model = get_embedding_model()
        
        # Generate embedding for the query
        query_embedding = model.encode(query).tolist()
        
        # Create filter for the specified papers
        if len(paper_ids) == 1:
            # If only one paper ID, use direct match
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_ids[0])
                    )
                ]
            )
        else:
            # If multiple paper IDs, use should (logical OR) with multiple conditions
            search_filter = Filter(
                should=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id)
                    ) for paper_id in paper_ids
                ]
            )
        
        # Search for similar chunks
        search_result = await client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=search_filter  # Changed from filter to query_filter
        )
        
        # Format the results
        contexts = []
        for result in search_result:
            contexts.append({
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "paper_id": result.payload["paper_id"],
                "chunk_index": result.payload["chunk_index"],
                "relevance_score": result.score
            })
        
        logger.info(f"Retrieved {len(contexts)} context chunks for query: '{query}'")
        return contexts
        
    except Exception as e:
        logger.error(f"Error retrieving context for query: {str(e)}", exc_info=True)
        raise

async def list_stored_papers(limit: int = 10, offset: int = 0, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List papers stored in the vector database.
    
    Args:
        limit: Maximum number of papers to return
        offset: Offset for pagination
        category: Optional category to filter by
        
    Returns:
        List of paper objects
    """
    try:
        collection_name = DEFAULT_COLLECTION_NAME
        client = await get_async_qdrant_client()
        
        # Create filter
        search_filter = Filter()
        if category:
            search_filter.must.append(
                FieldCondition(
                    key="metadata.categories",
                    match=MatchValue(
                        value=category
                    )
                )
            )
        
        # Use scroll to get all chunk IDs with paper_id
        result = await client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=1000,  # Get a reasonable number of chunks
            with_payload=True,
            with_vectors=False
        )
        
        # Extract unique paper IDs and their metadata
        paper_map = {}
        for point in result[0]:
            paper_id = point.payload["paper_id"]
            
            # Only store the first occurrence of each paper (to get complete metadata)
            if paper_id not in paper_map:
                paper_map[paper_id] = {
                    "id": paper_id,
                    "content": point.payload["content"],  # This will just be one chunk
                    "metadata": point.payload["metadata"]
                }
        
        # Convert to list and apply pagination
        papers = list(paper_map.values())
        paginated_papers = papers[offset:offset+limit]
        
        logger.info(f"Listed {len(paginated_papers)} papers (from total of {len(papers)})")
        return paginated_papers
        
    except Exception as e:
        logger.error(f"Error listing papers: {str(e)}", exc_info=True)
        raise

async def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a paper by ID.
    
    Args:
        paper_id: ID of the paper to retrieve
        
    Returns:
        Paper object or None if not found
    """
    try:
        collection_name = DEFAULT_COLLECTION_NAME
        client = await get_async_qdrant_client()
        
        # Create filter for the paper ID
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(
                        value=paper_id
                    )
                )
            ]
        )
        
        # Search for chunks with this paper ID
        result = await client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=100,  # Get a reasonable number of chunks
            with_payload=True,
            with_vectors=False
        )
        
        if not result[0]:
            logger.warning(f"Paper {paper_id} not found")
            return None
        
        # Reconstruct the full paper content from chunks
        chunks = sorted(
            result[0], 
            key=lambda point: point.payload.get("chunk_index", 0)
        )
        
        # Combine chunks into full content
        full_content = "\n".join([chunk.payload["content"] for chunk in chunks])
        
        # Get metadata from the first chunk
        metadata = chunks[0].payload["metadata"]
        
        paper = {
            "id": paper_id,
            "content": full_content,
            "metadata": metadata
        }
        
        logger.info(f"Retrieved paper {paper_id}")
        return paper
        
    except Exception as e:
        logger.error(f"Error retrieving paper: {str(e)}", exc_info=True)
        raise

async def delete_paper(paper_id: str) -> bool:
    """
    Delete a paper and all its chunks from the vector database.
    
    Args:
        paper_id: ID of the paper to delete
        
    Returns:
        True if successful, False if paper not found
    """
    try:
        collection_name = DEFAULT_COLLECTION_NAME
        client = await get_async_qdrant_client()
        
        # Create filter for the paper ID
        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(
                        value=paper_id
                    )
                )
            ]
        )
        
        # Delete all points with this paper ID
        result = await client.delete(
            collection_name=collection_name,
            points_selector=delete_filter
        )
        
        if result.status != "completed":
            logger.warning(f"Delete operation for paper {paper_id} failed")
            return False
        
        logger.info(f"Deleted paper {paper_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting paper: {str(e)}", exc_info=True)
        raise
