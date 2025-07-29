from fastapi import APIRouter, HTTPException
from app.models.hackrx import HackRXRequest, HackRXResponse
from app.services.document_processor import process_document_from_url
from app.services.embedding_service import create_embeddings, search_similar
from app.services.query_processor import process_queries
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/run", response_model=HackRXResponse)
async def run_hackrx_submission(request: HackRXRequest):
    """
    Main endpoint for HackRX submission - processes documents and answers queries
    """
    execution_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting HackRX execution {execution_id}")
        logger.info(f"Document URL: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Step 1: Process document from URL
        document_chunks = await process_document_from_url(request.documents)
        logger.info(f"Processed document into {len(document_chunks)} chunks")
        
        # Step 2: Create embeddings for document chunks
        await create_embeddings(document_chunks, execution_id)
        logger.info("Created embeddings for document chunks")
        
        # Step 3: Process each query
        answers = await process_queries(request.questions, execution_id)
        logger.info(f"Processed {len(answers)} queries")
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in HackRX execution {execution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "hackrx-query-retrieval",
        "timestamp": datetime.utcnow().isoformat()
    }