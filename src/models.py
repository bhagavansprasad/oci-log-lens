"""
models.py
---------
Pydantic models for OIC-LogLens REST API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── REQUEST MODELS ─────────────────────────────────────────────────────────────

class IngestFileRequest(BaseModel):
    """Request model for POST /ingest/file"""
    file_path: str = Field(..., description="Absolute path to the log file on the server")

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/flow-logs/01_flow-log.json"
            }
        }


# ── RESPONSE MODELS ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response model for all ingest endpoints"""
    log_id: str = Field(..., description="UUID of the ingested log record")
    jira_id: str = Field(..., description="Jira issue ID (extracted or generated)")
    status: str = Field(..., description="success or error")
    message: str = Field(..., description="Human readable message")

    class Config:
        json_schema_extra = {
            "example": {
                "log_id": "9f9da348-963c-41fe-8c61-3ec23dbb3c13",
                "jira_id": "https://promptlyai.atlassian.net/browse/OLL-4FF0674A",
                "status": "success",
                "message": "Log ingested successfully"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = Field(default="error", description="Always 'error'")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "File not found",
                "detail": "/path/to/log.json does not exist"
            }
        }



# ── SEARCH MODELS ──────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request model for POST /search"""
    log_content: str = Field(..., description="Raw log as JSON string (array format)")

    class Config:
        json_schema_extra = {
            "example": {
                "log_content": "[{...raw log json array...}]"
            }
        }


class SearchMatch(BaseModel):
    """Single search result match"""
    jira_id: str = Field(..., description="Jira issue ID")
    similarity_score: float = Field(..., description="Similarity percentage (0-100)")
    flow_code: str = Field(..., description="OIC flow code")
    trigger_type: Optional[str] = Field(None, description="Trigger type (rest/soap/scheduled)")
    error_code: Optional[str] = Field(None, description="Error code")
    error_summary: str = Field(..., description="Error summary (first 150 chars)")


class SearchResponse(BaseModel):
    """Response model for POST /search"""
    status: str = Field(..., description="success or error")
    message: str = Field(..., description="Human readable message")
    matches: list[SearchMatch] = Field(..., description="List of similar logs (Top-N)")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Found 5 similar logs",
                "matches": [
                    {
                        "jira_id": "https://promptlyai.atlassian.net/browse/OLL-4FF0674A",
                        "similarity_score": 100.0,
                        "flow_code": "RH_NAVAN_DAILY_INTEGR_SCHEDU",
                        "trigger_type": "scheduled",
                        "error_code": "Execution failed",
                        "error_summary": "oracle.cloud.connector.api.CloudInvocationException"
                    }
                ]
            }
        }



class IngestURLRequest(BaseModel):
    """Request model for POST /ingest/url"""
    url: str = Field(..., description="HTTP/HTTPS URL pointing to a log file")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/logs/error.json"
            }
        }


class IngestRawRequest(BaseModel):
    """Request model for POST /ingest/raw"""
    log_content: str = Field(..., description="Raw log as JSON string (array format)")

    class Config:
        json_schema_extra = {
            "example": {
                "log_content": "[{...raw log json array...}]"
            }
        }


class IngestDatabaseRequest(BaseModel):
    """Request model for POST /ingest/database"""
    connection_string: str = Field(..., description="Database connection string")
    query: str = Field(..., description="SQL query to fetch the log")

    class Config:
        json_schema_extra = {
            "example": {
                "connection_string": "oracle://user:pass@host:port/service",
                "query": "SELECT log_json FROM logs WHERE id = 123"
            }
        }



class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion"""
    status: str = Field(..., description="success or partial_success or error")
    message: str = Field(..., description="Human readable message")
    total_logs: int = Field(..., description="Total logs attempted")
    successful: int = Field(..., description="Successfully ingested logs")
    failed: int = Field(..., description="Failed logs")
    duplicates: int = Field(..., description="Duplicate logs")
    results: list[IngestResponse] = Field(default_factory=list, description="Individual results")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Batch ingestion completed",
                "total_logs": 5,
                "successful": 4,
                "failed": 0,
                "duplicates": 1,
                "results": []
            }
        }