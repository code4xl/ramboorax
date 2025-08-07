from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, EditAnswerRequest, BulkEditAnswersRequest
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


@router.post("/get-cached-qa")
async def get_cached_qa(req: QueryRequest):
    """Get cached question-answer pairs if they exist"""
    from app.helpers.cache_manager import generate_qa_cache_key, load_qa_cache_if_exists
    
    qa_cache_key = generate_qa_cache_key(req.documents, req.questions)
    cached_qa = load_qa_cache_if_exists(qa_cache_key)
    
    if cached_qa:
        # Map questions to answers maintaining input order
        cached_questions = cached_qa["questions"]
        cached_answers = cached_qa["answers"]
        qa_mapping = dict(zip(cached_questions, cached_answers))
        
        # Create response with question-answer pairs
        qa_pairs = []
        for i, question in enumerate(req.questions):
            answer = qa_mapping.get(question, None)
            qa_pairs.append({
                "question_index": i,
                "question": question,
                "answer": answer,
                "is_cached": answer is not None
            })
        
        return {
            "cache_found": True,
            "cache_key": qa_cache_key,
            "qa_pairs": qa_pairs
        }
    else:
        return {
            "cache_found": False,
            "message": "No cached answers found for this document and question set"
        }
    

@router.post("/edit-cached-answer")
async def edit_cached_answer(req: EditAnswerRequest):
    """Edit a specific answer in cached Q&A using exact question text"""
    from app.helpers.cache_manager import generate_qa_cache_key, load_qa_cache_if_exists, save_qa_cache
    
    qa_cache_key = generate_qa_cache_key(req.documents, req.questions)
    cached_qa = load_qa_cache_if_exists(qa_cache_key)
    
    if not cached_qa:
        raise HTTPException(status_code=404, detail="No cached answers found for this document and question set")
    
    # Find the target question in cached data
    cached_questions = cached_qa["questions"]
    cached_answers = cached_qa["answers"]
    
    try:
        question_index_in_cache = cached_questions.index(req.target_question)
    except ValueError:
        raise HTTPException(
            status_code=404, 
            detail=f"Question '{req.target_question}' not found in cache. Available questions: {cached_questions}"
        )
    
    # Update the specific answer at the correct position
    updated_answers = cached_answers.copy()
    old_answer = updated_answers[question_index_in_cache]
    updated_answers[question_index_in_cache] = req.new_answer
    
    # Save updated cache with original question order
    save_qa_cache(qa_cache_key, cached_questions, updated_answers)
    
    return {
        "success": True,
        "message": f"Answer updated successfully",
        "target_question": req.target_question,
        "old_answer": old_answer,
        "new_answer": req.new_answer,
        "cache_key": qa_cache_key
    }

@router.post("/delete-cached-qa")
async def delete_cached_qa(req: QueryRequest):
    """Delete cached question-answer pairs if they exist"""
    from app.helpers.cache_manager import generate_qa_cache_key, load_qa_cache_if_exists, delete_qa_cache
    
    qa_cache_key = generate_qa_cache_key(req.documents, req.questions)
    
    # Check if cache exists first
    cached_qa = load_qa_cache_if_exists(qa_cache_key)
    if not cached_qa:
        raise HTTPException(status_code=404, detail="No cached answers found for this document and question set")
    
    # Delete the cache
    deleted = delete_qa_cache(qa_cache_key)
    
    if deleted:
        return {
            "success": True,
            "message": "Cached Q&A deleted successfully",
            "deleted_cache_key": qa_cache_key,
            "deleted_questions_count": len(cached_qa["questions"]),
            "deleted_questions": cached_qa["questions"]
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to delete cache entry")
    
@router.post("/bulk-edit-cached-answers")
async def bulk_edit_cached_answers(req: BulkEditAnswersRequest):
    """Edit multiple answers in cached Q&A using the same sequence as questions"""
    from app.helpers.cache_manager import generate_qa_cache_key, load_qa_cache_if_exists, save_qa_cache
    
    # Validate input lengths match
    if len(req.questions) != len(req.newanswers):
        raise HTTPException(
            status_code=400, 
            detail=f"Questions count ({len(req.questions)}) doesn't match new answers count ({len(req.newanswers)})"
        )
    
    qa_cache_key = generate_qa_cache_key(req.documents, req.questions)
    cached_qa = load_qa_cache_if_exists(qa_cache_key)
    
    if not cached_qa:
        raise HTTPException(status_code=404, detail="No cached answers found for this document and question set")
    
    cached_questions = cached_qa["questions"]
    cached_answers = cached_qa["answers"]
    
    # Validate all questions exist in cache
    missing_questions = []
    question_indices = {}
    
    for i, question in enumerate(req.questions):
        try:
            cache_index = cached_questions.index(question)
            question_indices[i] = cache_index
        except ValueError:
            missing_questions.append(question)
    
    if missing_questions:
        raise HTTPException(
            status_code=404,
            detail=f"Questions not found in cache: {missing_questions}"
        )
    
    # Update answers
    updated_answers = cached_answers.copy()
    update_summary = []
    
    for i, new_answer in enumerate(req.newanswers):
        cache_index = question_indices[i]
        old_answer = updated_answers[cache_index]
        updated_answers[cache_index] = new_answer
        
        update_summary.append({
            "question": req.questions[i],
            "old_answer": old_answer,
            "new_answer": new_answer
        })
    
    # Save updated cache with original question order
    save_qa_cache(qa_cache_key, cached_questions, updated_answers)
    
    return {
        "success": True,
        "message": f"Successfully updated {len(req.newanswers)} answers",
        "cache_key": qa_cache_key,
        "updates_summary": update_summary,
        "total_questions_in_cache": len(cached_questions),
        "updated_count": len(req.newanswers)
    }