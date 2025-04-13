import os
import logging
import numpy as np
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embeddings(api_key: Optional[str] = None) -> Embeddings:
    """Initialize and return the embedding model."""
    if not api_key:
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "No Hugging Face API token found. Please set the HF_TOKEN environment variable "
                "or provide the api_key parameter."
            )
    
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise

def process_papers(documents: List[Document], api_key: Optional[str] = None, 
                   chunk_size: int = 1000, chunk_overlap: int = 200) -> Optional[FAISS]:
    """
    Process papers into chunks and create a vector store.
    
    Args:
        documents: List of Document objects containing paper content
        api_key: Hugging Face API token (optional if set as environment variable)
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        
    Returns:
        FAISS vector store or None if processing fails
    """
    if not documents:
        logger.warning("No documents provided for processing")
        return None
    
    try:
        # Initialize embeddings
        embeddings = get_embeddings(api_key)
        
        # Split documents into appropriate chunks for academic papers
        logger.info(f"Splitting documents with chunk size {chunk_size} and overlap {chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        # Log processing information
        logger.info(f"Processing {len(texts)} text chunks from {len(documents)} documents")
        if texts:
            logger.debug(f"Sample text: {texts[0].page_content[:100]}...")
        else:
            logger.warning("No text chunks generated from documents")
            return None
        
        # Create FAISS vector store directly from texts (simpler and more reliable)
        logger.info("Creating vector store from documents")
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            logger.info(f"Successfully created vector store with {len(texts)} entries")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store using from_documents: {str(e)}")
            
            # Fallback method using manual embedding
            logger.info("Attempting fallback method with manual embeddings")
            try:
                # Get text contents for embedding
                text_contents = [doc.page_content for doc in texts]
                metadatas = [doc.metadata for doc in texts]
                
                # Create the vector store
                vectorstore = FAISS.from_texts(
                    text_contents, 
                    embeddings, 
                    metadatas=metadatas
                )
                logger.info("Successfully created vector store using fallback method")
                return vectorstore
            except Exception as e2:
                logger.error(f"Fallback method also failed: {str(e2)}")
                return None
                
    except Exception as e:
        logger.error(f"Error in paper processing pipeline: {str(e)}")
        return None
