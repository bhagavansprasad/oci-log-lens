"""
main.py
-------
FastAPI REST API for OIC-LogLens.
Provides endpoints for log ingestion and deduplication search.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import IngestFileRequest, IngestResponse, SearchRequest, SearchResponse
from ingestion_service import load_from_file, load_from_raw_text, ingest_log
from search_service import search_log
from config import logger

# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OIC-LogLens API",
    description="AI-Powered Error Resolution Engine for Oracle Integration Cloud",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS Middleware ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "OIC-LogLens"}


# ── Ingest Endpoints ───────────────────────────────────────────────────────────

@app.post(
    "/ingest/file",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest log from file path"
)
def ingest_file(request: IngestFileRequest):
    """
    Ingest an OIC log from a file path on the server.
    
    The pipeline:
    1. Read file from provided path
    2. Normalize log using Gemini LLM
    3. Generate embedding vector
    4. Store in Oracle 26ai Vector Database
    
    Returns the LOG_ID of the ingested record.
    """
    # Load raw log from file
    raw_log = load_from_file(request.file_path)
    
    # Run ingestion pipeline
    log_id, jira_id = ingest_log(raw_log)
    
    return IngestResponse(
        log_id=log_id,
        jira_id=jira_id,
        status="success",
        message="Log ingested successfully"
    )



# ── Search Endpoint ────────────────────────────────────────────────────────────

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search for duplicate/similar logs"
)
def search_duplicate(request: SearchRequest):
    """
    Search for duplicate or similar logs using semantic similarity.
    
    The pipeline:
    1. Normalize query log using Gemini LLM
    2. Generate embedding vector
    3. Vector similarity search in Oracle 26ai
    4. Return Top-5 most similar logs
    
    Returns matches with Jira IDs, similarity scores, and metadata.
    """
    # Load raw log from text
    raw_log = load_from_raw_text(request.log_content)
    
    # Run search pipeline
    matches = search_log(raw_log, top_n=5)
    
    return SearchResponse(
        status="success",
        message=f"Found {len(matches)} similar logs",
        matches=matches
    )

# ── Run Server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )