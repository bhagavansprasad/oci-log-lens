"""
embedder.py
-----------
Embedding module for OIC-LogLens.
Builds a semantic text string from a normalized log and generates
vector embeddings using Gemini text-embedding-004.
"""

import logging
from google import genai
from prompts import get_embedding_text
from config import client, logger, EMBEDDING_MODEL

# ── EMBEDDER ───────────────────────────────────────────────────────────────────

def generate_embedding(normalized_log: dict) -> list[float]:
    """
    Generates a vector embedding from a normalized OIC log.

    Args:
        normalized_log: Normalized log dict (output of normalize_log).

    Returns:
        Embedding as a list of floats.

    Raises:
        ValueError: If the embedding text is empty.
        Exception:  If the Gemini API call fails.
    """

    # ── Step 1: Build embedding text from selected fields
    embedding_text = get_embedding_text(normalized_log)

    if not embedding_text.strip():
        raise ValueError("Embedding text is empty — check the normalized log fields.")

    logger.info(f"Embedding text: {embedding_text[:120]}...")

    # ── Step 2: Call Gemini embedding model
    logger.info(f"Generating embedding using {EMBEDDING_MODEL} ...")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=embedding_text
    )

    embedding = response.embeddings[0].values
    logger.info(f"Embedding generated. Dimensions: {len(embedding)}")

    return embedding