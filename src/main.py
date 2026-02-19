"""
main.py
-------
FastAPI REST API for OIC-LogLens.
Provides endpoints for log ingestion and deduplication search.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import (IngestFileRequest, IngestURLRequest, IngestRawRequest, 
                     IngestDatabaseRequest, IngestResponse, BatchIngestResponse,
                     SearchRequest, SearchResponse)
from ingestion_service import (load_from_file, load_from_url, load_from_raw_text, 
                                     load_from_database, ingest_log)
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



@app.post(
    "/ingest/url",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest log from URL"
)
def ingest_url(request: IngestURLRequest):
    """
    Ingest an OIC log from a URL.
    
    Fetches the log file from the provided HTTP/HTTPS URL,
    then runs the standard ingestion pipeline.
    """
    # Load raw log from URL
    raw_log = load_from_url(request.url)
    
    # Run ingestion pipeline
    log_id, jira_id = ingest_log(raw_log)
    
    return IngestResponse(
        log_id=log_id,
        jira_id=jira_id,
        status="success",
        message="Log ingested successfully"
    )


@app.post(
    "/ingest/raw",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest log from raw text"
)
def ingest_raw(request: IngestRawRequest):
    """
    Ingest an OIC log from raw JSON text.
    
    Accepts the log content as a JSON string,
    then runs the standard ingestion pipeline.
    """
    # Load raw log from text
    raw_log = load_from_raw_text(request.log_content)
    
    # Run ingestion pipeline
    log_id, jira_id = ingest_log(raw_log)
    
    return IngestResponse(
        log_id=log_id,
        jira_id=jira_id,
        status="success",
        message="Log ingested successfully"
    )


@app.post(
    "/ingest/database",
    response_model=BatchIngestResponse,
    tags=["Ingestion"],
    summary="Ingest logs from database (supports batch)"
)
def ingest_database(request: IngestDatabaseRequest):
    """
    Ingest OIC logs from a database query.
    
    Supports batch ingestion - if query returns multiple rows,
    all logs will be ingested and results returned.
    
    Returns summary with counts and individual results.
    """
    from fastapi import HTTPException, status
    
    # Load raw logs from database (may return multiple)
    raw_logs = load_from_database(request.connection_string, request.query)
    
    # Process each log
    results = []
    successful = 0
    failed = 0
    duplicates = 0
    
    for i, raw_log in enumerate(raw_logs, 1):
        try:
            logger.info(f"Processing log {i}/{len(raw_logs)}")
            log_id, jira_id = ingest_log(raw_log)
            
            results.append(IngestResponse(
                log_id=log_id,
                jira_id=jira_id,
                status="success",
                message=f"Log {i} ingested successfully"
            ))
            successful += 1
        
        except HTTPException as e:
            if e.status_code == 409:
                # Duplicate
                duplicates += 1
                results.append(IngestResponse(
                    log_id="",
                    jira_id="",
                    status="duplicate",
                    message=f"Log {i}: Duplicate detected"
                ))
            else:
                # Other error
                failed += 1
                results.append(IngestResponse(
                    log_id="",
                    jira_id="",
                    status="error",
                    message=f"Log {i}: {e.detail}"
                ))
        
        except Exception as e:
            failed += 1
            results.append(IngestResponse(
                log_id="",
                jira_id="",
                status="error",
                message=f"Log {i}: {str(e)}"
            ))
    
    # Determine overall status
    if successful == len(raw_logs):
        overall_status = "success"
    elif successful > 0:
        overall_status = "partial_success"
    else:
        overall_status = "error"
    
    return BatchIngestResponse(
        status=overall_status,
        message=f"Processed {len(raw_logs)} log(s): {successful} successful, {duplicates} duplicates, {failed} failed",
        total_logs=len(raw_logs),
        successful=successful,
        failed=failed,
        duplicates=duplicates,
        results=results
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