"""
ingestion_service.py
--------------------
Core ingestion pipeline and source loaders for OIC-LogLens.
Provides reusable functions for all ingestion endpoints.
"""

import json
import os
import hashlib
from typing import Dict, Any, List
from fastapi import HTTPException, status

from normalizer import normalize_log
from embedder import generate_embedding
from prompts import get_embedding_text
from db import insert_log, check_duplicate
from config import logger


# ── SOURCE LOADERS ─────────────────────────────────────────────────────────────

def load_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load raw log from a file path.
    
    Args:
        file_path: Absolute path to the log file
        
    Returns:
        Raw log as a list of dicts
        
    Raises:
        HTTPException: If file not found or invalid JSON
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}"
        )
    
    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a file: {file_path}"
        )
    
    logger.info(f"Reading file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            raw_log = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in file: {str(e)}"
        )
    
    if not isinstance(raw_log, list):
        logger.error("Log file must contain a JSON array")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Log file must contain a JSON array"
        )
    
    return raw_log


def load_from_url(url: str) -> List[Dict[str, Any]]:
    """
    Load raw log from a URL.
    
    Args:
        url: HTTP/HTTPS URL pointing to a log file
        
    Returns:
        Raw log as a list of dicts
        
    Raises:
        HTTPException: If URL unreachable or invalid JSON
    """
    # TODO: Implement URL fetching
    # import requests
    # response = requests.get(url, timeout=30)
    # response.raise_for_status()
    # raw_log = response.json()
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="URL ingestion not yet implemented"
    )


def load_from_raw_text(log_content: str) -> List[Dict[str, Any]]:
    """
    Load raw log from text/JSON string.
    
    Args:
        log_content: Raw log as JSON string
        
    Returns:
        Raw log as a list of dicts
        
    Raises:
        HTTPException: If invalid JSON
    """
    logger.info("Parsing raw log content")
    
    try:
        raw_log = json.loads(log_content)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in content: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in content: {str(e)}"
        )
    
    if not isinstance(raw_log, list):
        logger.error("Log content must be a JSON array")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Log content must be a JSON array"
        )
    
    return raw_log


def load_from_database(connection_string: str, query: str) -> List[Dict[str, Any]]:
    """
    Load raw log from a database query.
    
    Args:
        connection_string: Database connection string
        query: SQL query to fetch log
        
    Returns:
        Raw log as a list of dicts
        
    Raises:
        HTTPException: If connection fails or invalid result
    """
    # TODO: Implement database loading
    # import oracledb
    # conn = oracledb.connect(connection_string)
    # cursor = conn.cursor()
    # cursor.execute(query)
    # result = cursor.fetchone()
    # raw_log = json.loads(result[0])
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Database ingestion not yet implemented"
    )


# ── CORE INGESTION PIPELINE ────────────────────────────────────────────────────

def ingest_log(raw_log: List[Dict[str, Any]]) -> tuple[str, str]:
    """
    Core ingestion pipeline — shared by all ingestion endpoints.
    
    Pipeline:
    0. Check for duplicate (LOG_HASH)
    1. Normalize log using Gemini LLM
    2. Generate embedding vector
    3. Store in Oracle 26ai Vector Database
    
    Args:
        raw_log: Raw log as a list of dicts
        
    Returns:
        Tuple of (LOG_ID, JIRA_ID)
        
    Raises:
        HTTPException: If duplicate log or any step fails
    """
    try:
        # ── Step 0: Check for duplicate ────────────────────────────────────────
        raw_json_str = json.dumps(raw_log, sort_keys=True)
        log_hash = hashlib.sha256(raw_json_str.encode()).hexdigest()
        
        logger.info(f"Checking for duplicate log (hash: {log_hash[:16]}...)")
        
        if check_duplicate(log_hash):
            logger.warning(f"Duplicate log detected: {log_hash}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate log: This log has already been ingested (hash: {log_hash[:16]}...)"
            )
        
        # ── Step 1: Normalize ──────────────────────────────────────────────────
        logger.info("Normalizing log...")
        normalized_log = normalize_log(raw_log)
        
        # ── Step 1.5: Extract or generate Jira ID ──────────────────────────────
        jira_id = normalized_log.get("jira_id")
        
        if not jira_id:
            # Generate dummy Jira ID using first 8 chars of log_hash
            jira_ticket_id = f"OLL-{log_hash[:8].upper()}"
            jira_id = f"https://promptlyai.atlassian.net/browse/{jira_ticket_id}"
            logger.info(f"Generated Jira ID: {jira_id}")
        else:
            logger.info(f"Using extracted Jira ID: {jira_id}")
        
        # ── Step 2: Generate embedding ─────────────────────────────────────────
        logger.info("Generating embedding...")
        embedding = generate_embedding(normalized_log)
        semantic_text = get_embedding_text(normalized_log)
        
        # ── Step 3: Store in database ──────────────────────────────────────────
        logger.info("Storing in Oracle 26ai...")
        log_id = insert_log(
            normalized_log=normalized_log,
            raw_log=raw_log,
            embedding=embedding,
            semantic_text=semantic_text,
            jira_id=jira_id
        )
        
        logger.info(f"Ingestion successful. LOG_ID: {log_id}, JIRA_ID: {jira_id}")
        return log_id, jira_id
    
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )