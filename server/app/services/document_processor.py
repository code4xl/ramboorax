# import httpx
# import fitz
# import io
# import logging
# from typing import List, Dict
# import re

# logger = logging.getLogger(__name__)

# async def process_document_from_url(document_url: str) -> List[Dict]:
#     """
#     Download and process document from URL
#     """
#     try:
#         # Download document
#         async with httpx.AsyncClient() as client:
#             response = await client.get(str(document_url))
#             response.raise_for_status()
            
#         # Process based on content type
#         content_type = response.headers.get('content-type', '')
        
#         if 'pdf' in content_type.lower():
#             return await process_pdf(response.content)
#         elif 'word' in content_type.lower() or 'docx' in content_type.lower():
#             return await process_docx(response.content)
#         else:
#             # Try to process as PDF by default
#             return await process_pdf(response.content)
            
#     except Exception as e:
#         logger.error(f"Error processing document from URL: {str(e)}")
#         raise

# async def process_pdf(pdf_content: bytes) -> List[Dict]:
#     """
#     Extract text from PDF using pdfplumber
#     """
#     try:
#         pdf_file = io.BytesIO(pdf_content)
        
#         full_text = ""
#         with pdfplumber.open(pdf_file) as pdf:
#             for page in pdf.pages:
#                 text = page.extract_text()
#                 if text:
#                     full_text += text + "\n"
        
#         # Clean and chunk the text
#         chunks = intelligent_chunking(full_text)
        
#         return [{"text": chunk, "metadata": {"source": "pdf", "chunk_id": i}} 
#                 for i, chunk in enumerate(chunks)]
        
#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}")
#         raise
    
# def intelligent_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
#     """
#     Intelligent text chunking that preserves semantic meaning
#     """
#     # Clean text
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Split by sentences first
#     sentences = re.split(r'(?<=[.!?])\s+', text)
    
#     chunks = []
#     current_chunk = ""
    
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= max_chunk_size:
#             current_chunk += sentence + " "
#         else:
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "
    
#     if current_chunk:
#         chunks.append(current_chunk.strip())
    
#     return chunks

# async def process_docx(docx_content: bytes) -> List[Dict]:
#     """
#     Process DOCX files (implement based on your needs)
#     """
#     # Implement DOCX processing using python-docx
#     pass

import httpx
import fitz
import io
import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

async def process_document_from_url(document_url: str) -> List[Dict]:
    """
    Download and process document from URL
    """
    try:
        # Download document
        async with httpx.AsyncClient() as client:
            response = await client.get(str(document_url))
            response.raise_for_status()
            
        # Process based on content type
        content_type = response.headers.get('content-type', '')
        
        if 'pdf' in content_type.lower():
            return await process_pdf(response.content)
        elif 'word' in content_type.lower() or 'docx' in content_type.lower():
            return await process_docx(response.content)
        else:
            # Try to process as PDF by default
            return await process_pdf(response.content)
            
    except Exception as e:
        logger.error(f"Error processing document from URL: {str(e)}")
        raise

async def process_pdf(pdf_content: bytes) -> List[Dict]:
    """
    Extract text from PDF using PyMuPDF and chunk it intelligently
    """
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        full_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():  # Only add non-empty text
                full_text += text + "\n"
        
        pdf_document.close()
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Clean and chunk the text
        chunks = intelligent_chunking(full_text)
        
        return [{"text": chunk, "metadata": {"source": "pdf", "chunk_id": i}} 
                for i, chunk in enumerate(chunks)]
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def intelligent_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Intelligent text chunking that preserves semantic meaning
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max size and we have content
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If no chunks were created (text too short), return the original text
    if not chunks and text:
        chunks.append(text)
    
    return chunks

async def process_docx(docx_content: bytes) -> List[Dict]:
    """
    Process DOCX files using python-docx
    """
    try:
        from docx import Document
        import io
        
        # Create a file-like object from bytes
        docx_file = io.BytesIO(docx_content)
        doc = Document(docx_file)
        
        full_text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text += paragraph.text + "\n"
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from the DOCX")
        
        # Clean and chunk the text
        chunks = intelligent_chunking(full_text)
        
        return [{"text": chunk, "metadata": {"source": "docx", "chunk_id": i}} 
                for i, chunk in enumerate(chunks)]
        
    except ImportError:
        logger.error("python-docx not installed. Cannot process DOCX files.")
        raise ValueError("DOCX processing not available. Please install python-docx.")
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise