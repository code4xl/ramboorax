import fitz  # PyMuPDF for PDFs
import docx
import requests
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def extract_text_from_url(file_url: str) -> str:
    response = requests.get(file_url)
    ext = file_url.split('?')[0].split('.')[-1].lower()

    with NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
        f.write(response.content)
        f.flush()

        if ext == "pdf":
            doc = fitz.open(f.name)
            return "\n".join([page.get_text() for page in doc])

        elif ext == "docx":
            doc = docx.Document(f.name)
            return "\n".join([p.text for p in doc.paragraphs])

        elif ext == "eml":
            with open(f.name, "r", encoding="utf-8", errors="ignore") as email_file:
                html = email_file.read()
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n")

        else:
            return "❌ Unsupported file format"

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def chunk_text_parallel(text: str, chunk_size: int = 200, overlap: int = 50, num_threads: int = 4) -> list:
    """
    Split text into chunks using parallel processing for large documents
    """
    words = text.split()
    total_words = len(words)
    
    # If document is small, use regular chunking
    if total_words < 1000:
        return chunk_text(text, chunk_size, overlap)
    
    # Calculate words per thread
    words_per_thread = math.ceil(total_words / num_threads)
    
    def process_segment(start_idx, end_idx):
        segment_words = words[start_idx:end_idx]
        segment_text = " ".join(segment_words)
        return chunk_text(segment_text, chunk_size, overlap)
    
    # Create thread segments with overlap to maintain context
    segments = []
    for i in range(num_threads):
        start_idx = i * words_per_thread
        end_idx = min((i + 1) * words_per_thread + overlap, total_words)
        if start_idx < total_words:
            segments.append((start_idx, end_idx))
    
    # Process segments in parallel
    all_chunks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_segment, start, end) for start, end in segments]
        
        for future in concurrent.futures.as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)
    
    return all_chunks