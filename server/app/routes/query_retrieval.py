from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.document_processor import DocumentProcessorService
import time
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/run", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    processor_service = DocumentProcessorService()
    answers = await processor_service.process_document_and_questions(req.documents, req.questions)
    return QueryResponse(answers=answers)