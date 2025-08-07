from fastapi import FastAPI, Header, HTTPException, Request
from app.routes import query_retrieval, cache_management
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import json
import re
import os

load_dotenv()

class FixQuotesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        hackrx_post_endpoints = [
            "/api/v1/hackrx/run",
            "/api/v1/hackrx/get-cached-qa",
            "/api/v1/hackrx/edit-cached-answer",
            "/api/v1/hackrx/delete-cached-qa",
            "/api/v1/hackrx/bulk-edit-cached-answers"
        ]
        if request.url.path in hackrx_post_endpoints and request.method == "POST":
            # Read raw body
            body = await request.body()
            body_str = body.decode('utf-8')
            
            print(f"🔍 DEBUG: Original body: {body_str[:200]}...")
            
            try:
                # Try to parse as JSON
                json.loads(body_str)
                print("✅ DEBUG: Valid JSON, no fixing needed")
            except json.JSONDecodeError:
                print("🔧 DEBUG: Invalid JSON detected, attempting to fix...")
                
                # Fix the quotes issue
                fixed_body = self.fix_json_quotes(body_str)
                print(f"🔧 DEBUG: Fixed body: {fixed_body[:200]}...")
                
                # Replace the request body with fixed JSON
                request._body = fixed_body.encode('utf-8')
        
        response = await call_next(request)
        return response

    def fix_json_quotes(self, body_str: str) -> str:
        """Fix mixed or mismatched quotes in the questions array while preserving internal quotes."""
        try:
            pattern = r'"questions":\s*(\[.*?\])'
            match = re.search(pattern, body_str, re.DOTALL)
            if not match:
                return body_str

            questions_raw = match.group(1)
            print("🎯 DEBUG: Found questions array")

            questions = []
            i, n = 0, len(questions_raw)

            while i < n:
                if questions_raw[i] in ['"', "'"]:
                    start_quote = questions_raw[i]
                    i += 1
                    start_idx = i
                    question_chars = []
                    escaped = False

                    while i < n:
                        c = questions_raw[i]

                        if escaped:
                            question_chars.append(c)
                            escaped = False
                        elif c == '\\':
                            question_chars.append(c)
                            escaped = True
                        elif c == start_quote:
                            # Correct closing quote
                            break
                        elif c in ['"', "'"] and questions_raw[i+1:i+2] in [',', ']']:
                            # Potential mismatched closing at end
                            # Replace with start_quote later
                            break
                        else:
                            question_chars.append(c)
                        i += 1

                    # Construct question string
                    question = ''.join(question_chars)
                    end_quote = questions_raw[i] if i < n else start_quote

                    # Fix mismatched closing quotes
                    if end_quote != start_quote:
                        end_quote = start_quote

                    questions.append(f'{start_quote}{question}{end_quote}')
                    i += 1
                else:
                    i += 1

            print(f"📝 DEBUG: Extracted {len(questions)} questions")
            if questions:
                print(f"📝 DEBUG: First fixed question: {questions[0]}")

            # Rebuild valid JSON array with double quotes
            fixed_questions = []
            for q in questions:
                # Strip the wrapping quotes first, escape, then wrap again in double quotes
                inner = q[1:-1].replace('\\', '\\\\').replace('"', '\\"')
                fixed_questions.append(f'"{inner}"')

            fixed_array = '[' + ', '.join(fixed_questions) + ']'
            fixed_body = body_str.replace(questions_raw, fixed_array)
            return fixed_body

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
app.include_router(cache_management.router, prefix="/api/v1/cache", tags=["Cache Management"])

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

