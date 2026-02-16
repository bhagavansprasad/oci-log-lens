"""
normalizer.py
-------------
Normalization module for OIC-LogLens.
Sends a raw OIC log to Gemini and returns a normalized JSON object
matching the output contract defined in prompts.py.
"""

import json
import logging
from google import genai
from prompts import get_normalization_prompt

from config import client, logger, GENERATION_MODEL

def normalize_log(raw_log: list | str) -> dict:
    """
    Normalizes a raw OIC log using Gemini LLM.

    Args:
        raw_log: Raw OIC log as a Python list (JSON array) or a JSON string.

    Returns:
        Normalized log as a Python dict matching the output contract.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
        Exception:  If the Gemini API call fails.
    """

    if isinstance(raw_log, list):
        raw_log_str = json.dumps(raw_log, indent=2)
    else:
        raw_log_str = raw_log

    logger.info("Building normalization prompt ...")

    prompt = get_normalization_prompt(raw_log_str)

    logger.info(f"Sending log to Gemini ({GENERATION_MODEL}) for normalization ...")

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    raw_response = response.text.strip()
    logger.info("Response received from Gemini.")

    if raw_response.startswith("```"):
        lines = raw_response.splitlines()
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines[-1].strip() == "```" else lines
        raw_response = "\n".join(lines).strip()

    try:
        normalized = json.loads(raw_response)
        logger.info("Normalization successful.")
        return normalized
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response was:\n{raw_response}")
        raise ValueError(f"LLM did not return valid JSON: {e}") from e


def normalize_log_from_file(file_path: str) -> dict:
    """
    Convenience function — reads a raw OIC log file and normalizes it.

    Args:
        file_path: Path to the raw log JSON file.

    Returns:
        Normalized log as a Python dict matching the output contract.
    """
    logger.info(f"Reading log file: {file_path}")

    with open(file_path, "r") as f:
        raw_log = json.load(f)

    return normalize_log(raw_log)