"""
search_service.py
-----------------
Search and deduplication service for OIC-LogLens.
Provides semantic similarity search for duplicate detection.
"""

from typing import List, Dict, Any
from fastapi import HTTPException, status

from normalizer import normalize_log
from embedder import generate_embedding
from prompts import get_embedding_text
from db import search_similar_logs
from config import logger


# ── CORE SEARCH PIPELINE ───────────────────────────────────────────────────────

def search_log(raw_log: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Core search pipeline — semantic similarity search for duplicate detection.
    
    Pipeline:
    1. Normalize log using Gemini LLM
    2. Generate embedding vector
    3. Vector similarity search in Oracle 26ai
    4. Return Top-N matches with metadata
    
    Args:
        raw_log: Raw log as a list of dicts
        top_n: Number of top results to return (default: 5)
        
    Returns:
        List of search matches with jira_id, similarity_score, metadata
        
    Raises:
        HTTPException: If any step fails
    """
    try:
        # ── Step 1: Normalize ──────────────────────────────────────────────────
        logger.info("Normalizing query log...")
        normalized_log = normalize_log(raw_log)
        
        # ── Step 2: Generate embedding ─────────────────────────────────────────
        logger.info("Generating query embedding...")
        embedding = generate_embedding(normalized_log)
        
        # ── Step 3: Vector similarity search ───────────────────────────────────
        logger.info(f"Searching for Top-{top_n} similar logs...")
        results = search_similar_logs(embedding, top_n)
        
        # ── Step 4: Format results ─────────────────────────────────────────────
        matches = []
        for result in results:
            # Calculate similarity percentage from cosine distance
            distance = float(result.get("similarity_score", 1.0))
            similarity_score = round((1 - distance) * 100, 2)
            
            # Truncate error summary for display
            error_summary = result.get("error_summary") or ""
            error_summary_short = error_summary[:150] + ("..." if len(error_summary) > 150 else "")
            
            matches.append({
                "jira_id": result.get("jira_id"),
                "similarity_score": similarity_score,
                "flow_code": result.get("flow_code"),
                "trigger_type": result.get("trigger_type"),
                "error_code": result.get("error_code"),
                "error_summary": error_summary_short
            })
        
        logger.info(f"Search complete. {len(matches)} matches found.")
        return matches
    
    except Exception as e:
        logger.error(f"Search pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )