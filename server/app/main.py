from fastapi import FastAPI
from app.routes import query_retrieval
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

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

# Include routers
# app.include_router(document_processing.router, prefix="/api/documents", tags=["Document Processing"])
app.include_router(query_retrieval.router, prefix="/hackrx", tags=["Query Retrieval"])