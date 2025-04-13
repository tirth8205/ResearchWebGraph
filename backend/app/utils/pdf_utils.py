import os
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
import io
from datetime import datetime
import re

# Set up logging
logger = logging.getLogger(__name__)

# Try to import PDF processing libraries, with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    logger.warning("PyPDF2 not available. Attempting to install...")
    try:
        import subprocess
        subprocess.run(["pip", "install", "PyPDF2"], check=True)
        import PyPDF2
        PYPDF2_AVAILABLE = True
        logger.info("Successfully installed PyPDF2")
    except Exception as e:
        logger.error(f"Failed to install PyPDF2: {str(e)}")
        PYPDF2_AVAILABLE = False

# Try to import additional optional libraries for better PDF extraction
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.info("pdfminer.six not available. Using PyPDF2 for text extraction.")

async def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from a PDF file asynchronously.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Extracted text as a string
    """
    # Run in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(
        _extract_text_from_pdf_sync,
        file_content
    )

def _extract_text_from_pdf_sync(file_content: bytes) -> str:
    """
    Synchronous implementation of extract_text_from_pdf.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Extracted text as a string
    """
    if not PYPDF2_AVAILABLE:
        raise RuntimeError("PyPDF2 is required for PDF text extraction but is not available")
    
    try:
        # Try PDF extraction with PyPDF2 first
        text = _extract_with_pypdf2(file_content)
        
        # If PyPDF2 extraction yields no text, try PDFMiner if available
        if not text and PDFMINER_AVAILABLE:
            logger.info("PyPDF2 extraction yielded no text, trying PDFMiner")
            text = _extract_with_pdfminer(file_content)
        
        # Clean up the extracted text
        text = _clean_extracted_text(text)
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

def _extract_with_pypdf2(file_content: bytes) -> str:
    """
    Extract text from PDF using PyPDF2.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Extracted text as a string
    """
    text = ""
    
    try:
        # Create a file-like object from the bytes
        pdf_stream = io.BytesIO(file_content)
        
        # Use PyPDF2 to read the PDF
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        # Check if the PDF is encrypted
        if pdf_reader.is_encrypted:
            try:
                # Try with empty password
                pdf_reader.decrypt('')
            except:
                logger.warning("PDF is encrypted and could not be decrypted")
                return "This PDF is encrypted and could not be processed."
        
        # Extract metadata if available
        metadata = {}
        if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
            for key, value in pdf_reader.metadata.items():
                if isinstance(value, str):
                    metadata[key] = value
        
        # Extract text from each page
        pages = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text:
                pages.append(page_text)
            else:
                logger.warning(f"No text extracted from page {page_num+1}")
                pages.append(f"[Page {page_num+1} - No text could be extracted]")
        
        # Join all pages with page markers
        text = "\n\n--- Page Break ---\n\n".join(pages)
        
        return text
    
    except Exception as e:
        logger.error(f"Error in PyPDF2 extraction: {str(e)}", exc_info=True)
        raise

def _extract_with_pdfminer(file_content: bytes) -> str:
    """
    Extract text from PDF using PDFMiner.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Extracted text as a string
    """
    if not PDFMINER_AVAILABLE:
        return ""
    
    try:
        # Create a file-like object from the bytes
        pdf_stream = io.BytesIO(file_content)
        
        # Extract text with PDFMiner
        text = pdfminer_extract_text(pdf_stream)
        return text
    
    except Exception as e:
        logger.error(f"Error in PDFMiner extraction: {str(e)}", exc_info=True)
        return ""

def _clean_extracted_text(text: str) -> str:
    """
    Clean up extracted text, removing unnecessary whitespace and formatting issues.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove unnecessary Unicode characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Remove repeated page numbers or headers/footers (common in academic papers)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text.strip()

async def extract_pdf_metadata(file_content: bytes) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file asynchronously.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Dictionary of metadata
    """
    # Run in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(
        _extract_pdf_metadata_sync,
        file_content
    )

def _extract_pdf_metadata_sync(file_content: bytes) -> Dict[str, Any]:
    """
    Synchronous implementation of extract_pdf_metadata.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Dictionary of metadata
    """
    if not PYPDF2_AVAILABLE:
        raise RuntimeError("PyPDF2 is required for PDF metadata extraction but is not available")
    
    try:
        # Create a file-like object from the bytes
        pdf_stream = io.BytesIO(file_content)
        
        # Use PyPDF2 to read the PDF
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        # Initialize metadata dictionary
        metadata = {
            "page_count": len(pdf_reader.pages),
            "extraction_date": datetime.now().isoformat()
        }
        
        # Extract standard metadata fields
        if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
            pdf_info = pdf_reader.metadata
            
            # Map PDF metadata to our schema
            field_mapping = {
                "/Title": "title",
                "/Author": "author",
                "/Subject": "subject",
                "/Keywords": "keywords",
                "/Producer": "producer",
                "/Creator": "creator",
                "/CreationDate": "creation_date",
                "/ModDate": "modification_date"
            }
            
            for pdf_field, our_field in field_mapping.items():
                if pdf_field in pdf_info:
                    value = pdf_info[pdf_field]
                    
                    # Clean up dates
                    if "date" in our_field.lower() and value.startswith("D:"):
                        # Convert PDF date format
                        value = _parse_pdf_date(value)
                    
                    metadata[our_field] = value
        
        # Try to extract title from first page if not in metadata
        if "title" not in metadata:
            try:
                first_page_text = pdf_reader.pages[0].extract_text()
                lines = first_page_text.strip().split('\n')
                # Assume the first non-empty line could be the title
                if lines and len(lines[0]) > 10:  # A reasonable title length
                    metadata["title"] = lines[0].strip()
            except:
                pass
        
        # Try to extract authors from first page if not in metadata
        if "author" not in metadata:
            try:
                first_page_text = pdf_reader.pages[0].extract_text()
                # Look for typical author patterns in academic papers
                author_match = re.search(r'(?:Author|Authors|By)[s:]*\s+((?:[A-Z][a-z]+ [A-Z][a-z]+(?:,? (?:and |& ))?)+)', first_page_text)
                if author_match:
                    metadata["author"] = author_match.group(1).strip()
            except:
                pass
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error extracting PDF metadata: {str(e)}", exc_info=True)
        # Return minimal metadata on error
        return {
            "extraction_error": str(e),
            "extraction_date": datetime.now().isoformat()
        }

def _parse_pdf_date(date_str: str) -> str:
    """
    Parse PDF date format to ISO format.
    
    Args:
        date_str: PDF date string (e.g., 'D:20210527123456+02'00'')
        
    Returns:
        ISO format date string
    """
    try:
        # Remove 'D:' prefix
        if date_str.startswith('D:'):
            date_str = date_str[2:]
        
        # Basic format: YYYYMMDDHHMMSS
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        
        # Optional time components
        hour = "00"
        minute = "00"
        second = "00"
        
        if len(date_str) >= 10:
            hour = date_str[8:10]
        if len(date_str) >= 12:
            minute = date_str[10:12]
        if len(date_str) >= 14:
            second = date_str[12:14]
        
        # Format as ISO
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    
    except Exception:
        # Return original string if parsing fails
        return date_str

async def is_valid_pdf(file_content: bytes) -> Tuple[bool, Optional[str]]:
    """
    Check if a file is a valid PDF.
    
    Args:
        file_content: File content as bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Run in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(
        _is_valid_pdf_sync,
        file_content
    )

def _is_valid_pdf_sync(file_content: bytes) -> Tuple[bool, Optional[str]]:
    """
    Synchronous implementation of is_valid_pdf.
    
    Args:
        file_content: File content as bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not PYPDF2_AVAILABLE:
        return False, "PDF validation requires PyPDF2 which is not available"
    
    try:
        # Check if the file starts with the PDF signature
        if not file_content.startswith(b'%PDF'):
            return False, "File does not start with PDF signature"
        
        # Create a file-like object from the bytes
        pdf_stream = io.BytesIO(file_content)
        
        # Try to read the PDF
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        # Check if there are pages
        if len(pdf_reader.pages) == 0:
            return False, "PDF contains no pages"
        
        # Check if encrypted
        if pdf_reader.is_encrypted:
            try:
                # Try with empty password
                if not pdf_reader.decrypt(''):
                    return False, "PDF is encrypted and could not be decrypted"
            except:
                return False, "PDF is encrypted and could not be decrypted"
        
        # PDF is valid
        return True, None
    
    except Exception as e:
        return False, f"Invalid PDF file: {str(e)}"

