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