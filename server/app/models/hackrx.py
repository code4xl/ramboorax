from pydantic import BaseModel, HttpUrl
from typing import List

class HackRXRequest(BaseModel):
    """HackRX API request model"""
    documents: HttpUrl
    questions: List[str]

class HackRXResponse(BaseModel):
    """HackRX API response model"""
    answers: List[str]