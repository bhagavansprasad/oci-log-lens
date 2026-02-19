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
from prompts import get_embedding_text, get_rerank_prompt
from db import search_similar_logs
from config import logger



# ── LLM RE-RANKING ─────────────────────────────────────────────────────────────

def rerank_with_llm(normalized_log: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use LLM to re-rank and classify search results.
    
    Args:
        normalized_log: The normalized query log
        candidates: List of candidate matches from vector search
        
    Returns:
        Re-ranked and classified list of matches
    """
    import json
    from config import client, GENERATION_MODEL
    
    logger.info(f"Re-ranking {len(candidates)} candidates with LLM...")
    
    try:
        # Get re-ranking prompt with schema
        system_prompt, user_prompt, response_schema = get_rerank_prompt(normalized_log, candidates)
        
        # Call Gemini with structured output
        logger.info("Calling Gemini for re-ranking...")
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[system_prompt, user_prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )
        
        # Parse structured response
        reranked_text = response.text.strip()
        response_data = json.loads(reranked_text)
        reranked_results = response_data.get("results", [])
        logger.info(f"Re-ranking complete: {len(reranked_results)} results classified")
        
        # Merge re-ranking data with original candidates
        # Create lookup by both full URL and short ticket ID
        jira_to_candidate = {}
        for c in candidates:
            full_url = c.get("jira_id")
            jira_to_candidate[full_url] = c
            # Also map short ID (e.g., OLL-3C5C205C)
            if full_url:
                short_id = full_url.split("/")[-1]
                jira_to_candidate[short_id] = c
        
        enhanced_results = []
        for result in reranked_results:
            jira_id = result.get("jira_id")
            if jira_id in jira_to_candidate:
                candidate = jira_to_candidate[jira_id].copy()
                candidate.update({
                    "rank": result.get("rank"),
                    "classification": result.get("classification"),
                    "confidence": result.get("confidence"),
                    "reasoning": result.get("reasoning")
                })
                enhanced_results.append(candidate)
            else:
                logger.warning(f"Jira ID {jira_id} from LLM not found in candidates")
        
        # Sort by rank
        enhanced_results.sort(key=lambda x: x.get("rank", 999))
        
        logger.info(f"Re-ranking complete: {len(enhanced_results)} results enhanced")
        return enhanced_results
    
    except Exception as e:
        logger.warning(f"Re-ranking failed: {e}. Returning original results.")
        # If re-ranking fails, return original results
        return candidates


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
        logger.info(f"Vector search returned {len(results)} results")
        
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
                "error_summary": error_summary_short,
                "normalized_json": result.get("normalized_json", {})  # Add for re-ranking
            })
        
        # ── Step 5: LLM Re-ranking ─────────────────────────────────────────────
        logger.info("Re-ranking results with LLM for smarter duplicate detection...")
        enhanced_matches = rerank_with_llm(normalized_log, matches)
        
        # Return top_n after re-ranking
        final_results = enhanced_matches[:top_n]
        
        logger.info(f"Search complete. {len(final_results)} matches returned after re-ranking.")
        return final_results
    
    except Exception as e:
        logger.error(f"Search pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
