"""
config.py
---------
Shared configuration for OIC-LogLens.
Import from here — do not redefine in individual modules.
"""

import logging
from google import genai

# ── LOGGING ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("oic_loglens")

# ── MODELS ─────────────────────────────────────────────────────────────────────

GENERATION_MODEL  = "gemini-2.0-flash"
EMBEDDING_MODEL   = "text-embedding-004"

# ── CLIENT ─────────────────────────────────────────────────────────────────────

client = genai.Client()