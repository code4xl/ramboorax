import fitz  # PyMuPDF for PDFs
import docx
import requests
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

#for various File Types:
import openpyxl
from PIL import Image
import pytesseract
from pptx import Presentation
import zipfile
import io
from urllib.parse import urlparse
import mimetypes
import os

# def extract_text_from_url(file_url: str) -> str:
#     response = requests.get(file_url)
#     ext = file_url.split('?')[0].split('.')[-1].lower()

#     with NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
#         f.write(response.content)
#         f.flush()

#         if ext == "pdf":
#             doc = fitz.open(f.name)
#             return "\n".join([page.get_text() for page in doc])

#         elif ext == "docx":
#             doc = docx.Document(f.name)
#             return "\n".join([p.text for p in doc.paragraphs])

#         elif ext == "eml":
#             with open(f.name, "r", encoding="utf-8", errors="ignore") as email_file:
#                 html = email_file.read()
#                 soup = BeautifulSoup(html, "html.parser")
#                 return soup.get_text(separator="\n")

#         else:
#             return "❌ Unsupported file format"

def extract_text_from_url(file_url: str) -> dict:
    """Extract text from various file formats with security checks"""
    
    # Check for fraudulent URLs
    if is_fraudulent_url(file_url):
        return {
            "isError": True,
            "message": "Suspicious or fraudulent URL detected. File processing blocked for security reasons."
        }
    
    try:
        # Get file info without downloading
        head_response = requests.head(file_url, timeout=10)
        file_size = get_file_size(head_response)
        
        # Check file size (1GB limit)
        if file_size > 1024 * 1024 * 1024:  # 1GB in bytes
            return {
                "isError": True,
                "message": f"File too large ({file_size / (1024*1024*1024):.2f} GB). Maximum allowed size is 1GB."
            }
        
        # Download with streaming and size check
        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Double-check file size during download
        downloaded_size = 0
        content_chunks = []
        
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > 1024 * 1024 * 1024:  # 1GB limit
                return {
                    "isError": True,
                    "message": "File size exceeded during download. Processing stopped for security reasons."
                }
            content_chunks.append(chunk)
        
        file_content = b''.join(content_chunks)
        
        # Determine file extension
        ext = file_url.split('?')[0].split('.')[-1].lower() if '.' in file_url else ''
        
        # If no extension in URL, try to detect from content-type
        if not ext:
            content_type = response.headers.get('content-type', '').lower()
            ext_map = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
                'image/jpeg': 'jpg',
                'image/png': 'png',
                'application/zip': 'zip',
                'text/html': 'eml'
            }
            ext = ext_map.get(content_type, '')
        
        if not ext:
            return {
                "isError": True,
                "message": "Unable to determine file type. Please ensure the file has a valid extension."
            }
        
        # Create temporary file
        with NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
            f.write(file_content)
            f.flush()
            
            try:
                extracted_text = ""
                
                if ext == "pdf":
                    doc = fitz.open(f.name)
                    extracted_text = "\n".join([page.get_text() for page in doc])

                elif ext == "docx":
                    doc = docx.Document(f.name)
                    extracted_text = "\n".join([p.text for p in doc.paragraphs])

                elif ext == "xlsx":
                    extracted_text = extract_text_from_xlsx(f.name)

                elif ext in ["jpg", "jpeg", "png"]:
                    extracted_text = extract_text_from_image(f.name)

                elif ext == "pptx":
                    extracted_text = extract_text_from_pptx(f.name)

                elif ext == "zip":
                    extracted_text = extract_text_from_zip(f.name)

                elif ext == "eml":
                    with open(f.name, "r", encoding="utf-8", errors="ignore") as email_file:
                        html = email_file.read()
                        soup = BeautifulSoup(html, "html.parser")
                        extracted_text = soup.get_text(separator="\n")

                else:
                    return {
                        "isError": True,
                        "message": f"Unsupported file format: .{ext}. Supported formats: PDF, DOCX, XLSX, JPG, PNG, PPTX, ZIP, EML"
                    }
                
                # Check if extraction was successful
                if not extracted_text or extracted_text.strip() == "":
                    return {
                        "isError": True,
                        "message": f"No content could be extracted from the {ext.upper()} file. The file might be empty or corrupted."
                    }
                
                return {
                    "isError": False,
                    "text": extracted_text
                }
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(f.name)
                except:
                    pass
    
    except requests.exceptions.Timeout:
        return {
            "isError": True,
            "message": "Request timeout. The file took too long to download. Please try again or use a different file."
        }
    except requests.exceptions.RequestException as e:
        return {
            "isError": True,
            "message": f"Network error occurred while downloading the file: {str(e)}"
        }
    except Exception as e:
        return {
            "isError": True,
            "message": f"Unexpected error occurred while processing the file: {str(e)}"
        }

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Improved chunking with sentence boundary preservation and better overlap
    """
    import re
    
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed chunk_size, create a new chunk
        if current_word_count + sentence_words > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep overlap sentences for context
            overlap_words = 0
            overlap_sentences = []
            for i in range(len(current_chunk) - 1, -1, -1):
                sentence_word_count = len(current_chunk[i].split())
                if overlap_words + sentence_word_count <= overlap:
                    overlap_sentences.insert(0, current_chunk[i])
                    overlap_words += sentence_word_count
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_word_count = overlap_words + sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunk_text_parallel(text: str, chunk_size: int = 500, overlap: int = 100, num_threads: int = 4) -> list:
    """
    Split text into chunks using parallel processing for large documents with improved chunking
    """
    words = text.split()
    total_words = len(words)
    
    # If document is small, use regular chunking
    if total_words < 2000:
        return chunk_text(text, chunk_size, overlap)
    
    # Calculate words per thread with larger segments
    words_per_thread = math.ceil(total_words / num_threads)
    
    def process_segment(start_idx, end_idx):
        segment_words = words[start_idx:end_idx]
        segment_text = " ".join(segment_words)
        return chunk_text(segment_text, chunk_size, overlap)
    
    # Create thread segments with overlap to maintain context
    segments = []
    for i in range(num_threads):
        start_idx = i * words_per_thread
        end_idx = min((i + 1) * words_per_thread + overlap * 2, total_words)  # Increased overlap
        if start_idx < total_words:
            segments.append((start_idx, end_idx))
    
    # Process segments in parallel
    all_chunks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_segment, start, end) for start, end in segments]
        
        for future in concurrent.futures.as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)
    
    # Remove near-duplicate chunks that might result from overlapping
    deduplicated_chunks = []
    for i, chunk in enumerate(all_chunks):
        is_duplicate = False
        for existing_chunk in deduplicated_chunks:
            # Check for significant overlap (>70%)
            chunk_words = set(chunk.lower().split())
            existing_words = set(existing_chunk.lower().split())
            overlap_ratio = len(chunk_words.intersection(existing_words)) / max(len(chunk_words), len(existing_words))
            if overlap_ratio > 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated_chunks.append(chunk)
    
    return deduplicated_chunks



def is_fraudulent_url(url: str) -> bool:
    """Check if URL appears to be fraudulent or suspicious"""
    suspicious_patterns = [
        '.bin', '.exe', '.bat', '.cmd', '.scr', '.com', '.pif',
        'download', 'temp', 'trash', 'dummy', 'test-file',
        'speed-test', 'bandwidth', 'hetzner.com', 'speedtest'
    ]
    
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in suspicious_patterns)

def get_file_size(response) -> int:
    """Get file size from response headers"""
    content_length = response.headers.get('content-length')
    if content_length:
        return int(content_length)
    return 0

def extract_text_from_xlsx(file_path: str) -> str:
    """Extract text from Excel file"""
    try:
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        text_content = []
        
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            text_content.append(f"Sheet: {sheet_name}\n")
            
            for row in sheet.iter_rows(values_only=True):
                row_text = []
                for cell in row:
                    if cell is not None:
                        row_text.append(str(cell))
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        result = "\n".join(text_content)
        return result if result.strip() else "Excel file appears to be empty"
    except Exception as e:
        raise Exception(f"Failed to read Excel file: {str(e)}")

def extract_text_from_image(file_path: str) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else "No text found in the image"
    except Exception as e:
        raise Exception(f"Failed to process image file: {str(e)}")

def extract_text_from_pptx(file_path: str) -> str:
    """Extract text from PowerPoint file"""
    try:
        presentation = Presentation(file_path)
        text_content = []
        
        for i, slide in enumerate(presentation.slides, 1):
            text_content.append(f"Slide {i}:")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text.strip())
            text_content.append("")  # Empty line between slides
        
        result = "\n".join(text_content)
        return result if result.strip() else "PowerPoint file appears to contain no text"
    except Exception as e:
        raise Exception(f"Failed to read PowerPoint file: {str(e)}")

def extract_text_from_zip(file_path: str) -> str:
    """Extract text from ZIP file (list contents and extract text files)"""
    try:
        text_content = ["ZIP Archive Contents:"]
        
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            text_content.extend([f"- {filename}" for filename in file_list[:50]])  # Limit to 50 files
            
            if len(file_list) > 50:
                text_content.append(f"... and {len(file_list) - 50} more files")
            
            # Try to extract text from .txt files in the ZIP
            text_content.append("\nText file contents:")
            for filename in file_list[:10]:  # Only process first 10 files
                if filename.lower().endswith(('.txt', '.md', '.csv')):
                    try:
                        with zip_file.open(filename) as file:
                            content = file.read().decode('utf-8', errors='ignore')
                            text_content.append(f"\n--- {filename} ---")
                            text_content.append(content[:1000])  # First 1000 chars only
                    except:
                        continue
        
        return "\n".join(text_content)
    except Exception as e:
        raise Exception(f"Failed to read ZIP file: {str(e)}")