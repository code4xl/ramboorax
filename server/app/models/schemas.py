from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class EditAnswerRequest(BaseModel):
    documents: str
    questions: List[str]  # Keep this for cache key generation
    target_question: str  # The exact question text to update
    new_answer: str       # New answer text