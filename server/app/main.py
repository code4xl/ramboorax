from fastapi import FastAPI
from app.routes import workflow
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GitHub Stats API",
    docs_url="/aai",  # Custom docs URL
    redoc_url="/reaai"  # Custom redoc URL
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow.router, prefix="/api", tags=["Workflow Execution"])
