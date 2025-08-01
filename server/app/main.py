from fastapi import FastAPI, Header, HTTPException, Request
from app.routes import query_retrieval
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import json
import re
import os

load_dotenv()

class FixQuotesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/api/v1/hackrx/run" and request.method == "POST":
            # Read the raw body
            body = await request.body()
            body_str = body.decode('utf-8')
            
            print(f"🔍 DEBUG: Original body: {body_str[:200]}...")
            
            try:
                # Try to parse as normal JSON first
                json.loads(body_str)
                print("✅ DEBUG: Valid JSON, no fixing needed")
            except json.JSONDecodeError:
                print("🔧 DEBUG: Invalid JSON detected, attempting to fix...")
                
                # Fix the quotes issue
                fixed_body = self.fix_json_quotes(body_str)
                print(f"🔧 DEBUG: Fixed body: {fixed_body[:200]}...")
                
                # Replace the request body
                request._body = fixed_body.encode('utf-8')
        
        response = await call_next(request)
        return response
    
    def fix_json_quotes(self, body_str: str) -> str:
        """Fix mixed quotes in JSON string"""
        try:
            # Find the questions array using regex
            pattern = r'"questions":\s*(\[.*?\])'
            match = re.search(pattern, body_str, re.DOTALL)
            
            if match:
                questions_array = match.group(1)
                print(f"🎯 DEBUG: Found questions array")
                
                # Parse each question individually based on its wrapper quotes
                questions = []
                
                # Find all question strings - look for patterns that start with either " or '
                i = 0
                while i < len(questions_array):
                    # Skip whitespace, commas, brackets
                    if questions_array[i] in ' \n\t,[]':
                        i += 1
                        continue
                    
                    # Found start of a question
                    if questions_array[i] == '"':
                        # Double-quoted string - find the closing quote
                        i += 1  # skip opening quote
                        start = i
                        while i < len(questions_array) and questions_array[i] != '"':
                            if questions_array[i] == '\\':  # handle escaped characters
                                i += 2
                            else:
                                i += 1
                        
                        if i < len(questions_array):  # found closing quote
                            question = questions_array[start:i]
                            # Remove all single quotes from double-quoted string
                            clean_question = question.replace("'", "")
                            questions.append(clean_question)
                            i += 1  # skip closing quote
                        
                    elif questions_array[i] == "'":
                        # Single-quoted string - find the closing quote
                        i += 1  # skip opening quote
                        start = i
                        while i < len(questions_array) and questions_array[i] != "'":
                            if questions_array[i] == '\\':  # handle escaped characters
                                i += 2
                            else:
                                i += 1
                        
                        if i < len(questions_array):  # found closing quote
                            question = questions_array[start:i]
                            # Remove all double quotes from single-quoted string
                            clean_question = question.replace('"', "")
                            questions.append(clean_question)
                            i += 1  # skip closing quote
                    else:
                        i += 1
                
                print(f"📝 DEBUG: Extracted {len(questions)} questions")
                print(f"📝 DEBUG: First question: {questions[0] if questions else 'None'}")
                
                # Rebuild as proper JSON array with double quotes
                fixed_questions = []
                for q in questions:
                    # Escape any remaining double quotes and backslashes for JSON
                    escaped_q = q.replace('\\', '\\\\').replace('"', '\\"')
                    fixed_questions.append(f'"{escaped_q}"')
                
                fixed_array = '[' + ', '.join(fixed_questions) + ']'
                
                # Replace in the original body
                fixed_body = body_str.replace(match.group(1), fixed_array)
                return fixed_body
            
            return body_str
            
        except Exception as e:
            print(f"❌ DEBUG: Error fixing quotes: {e}")
            return body_str

app = FastAPI(
    title="LLM-Powered Query Retrieval System",
    description="Intelligent document processing and query system for insurance, legal, and compliance domains",
    version="1.0.0",
    docs_url="/aai",
    redoc_url="/reaai"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add the quote fixing middleware
app.add_middleware(FixQuotesMiddleware)

app.include_router(query_retrieval.router, prefix="/api/v1/hackrx", tags=["Query Retrieval"])

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "LLM-Powered Query Retrieval System",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Google Cloud Run"""
    return {"status": "healthy"}

