"""
db.py
-----
Database module for OIC-LogLens.
Handles all interactions with Oracle 26ai — inserts normalized logs,
raw logs, embeddings, and Jira IDs into the OLL_LOGS table.
"""

import json
import uuid
import array
import oracledb
from datetime import datetime
from config import logger

# ── CONNECTION CONFIG ──────────────────────────────────────────────────────────
# Update these values to match your Oracle 26ai environment.

DB_USER     = "EA_APP"
DB_PASSWORD = "jnjnuh"
DB_DSN      = "localhost/FREEPDB1"

# ── SQL ────────────────────────────────────────────────────────────────────────

INSERT_LOG_SQL = """
INSERT INTO OLL_LOGS (
    LOG_ID,
    JIRA_ID,
    LOG_TYPE,
    EVENT_TIME,
    FLOW_CODE,
    TRIGGER_TYPE,
    ENDPOINT_NAME,
    ERROR_CODE,
    ERROR_SUMMARY,
    SEMANTIC_TEXT,
    RAW_JSON,
    NORMALIZED_JSON,
    VECTOR
) VALUES (
    :log_id,
    :jira_id,
    :log_type,
    :event_time,
    :flow_code,
    :trigger_type,
    :endpoint_name,
    :error_code,
    :error_summary,
    :semantic_text,
    :raw_json,
    :normalized_json,
    :vector
)
"""


# ── CONNECTION ─────────────────────────────────────────────────────────────────

def get_connection() -> oracledb.Connection:
    """
    Creates and returns a new Oracle DB connection.

    Returns:
        oracledb.Connection instance.
    """
    logger.info("Connecting to Oracle 26ai ...")
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)
    logger.info("Connected.")
    return conn


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _parse_event_time(timestamp_str: str | None) -> datetime | None:
    """
    Parses ISO 8601 timestamp string into a Python datetime object.

    Args:
        timestamp_str: ISO 8601 timestamp string or None.

    Returns:
        datetime object or None.
    """
    if not timestamp_str:
        return None
    try:
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None


def _to_vector_array(embedding: list[float]) -> array.array:
    """
    Converts a list of floats into a FLOAT32 array for Oracle VECTOR column.

    Args:
        embedding: List of floats from the embedding model.

    Returns:
        array.array of type float32.
    """
    return array.array("f", embedding)


def _build_record(
    normalized_log: dict,
    raw_log: list,
    embedding: list[float],
    semantic_text: str,
    jira_id: str | None
) -> dict:
    """
    Builds the database record dict from normalized log, raw log, and embedding.

    Args:
        normalized_log:  Normalized log dict (output of normalize_log).
        raw_log:         Original raw log list (JSON array).
        embedding:       Vector embedding (output of generate_embedding).
        semantic_text:   Text used to generate the embedding.
        jira_id:         Associated Jira issue ID (optional).

    Returns:
        Dict of column name → value ready for INSERT.
    """
    flow  = normalized_log.get("flow")  or {}
    error = normalized_log.get("error") or {}
    msg   = error.get("message_parsed") or {}

    return {
        "log_id":          str(uuid.uuid4()),
        "jira_id":         jira_id,
        "log_type":        normalized_log.get("log_type"),
        "event_time":      _parse_event_time(flow.get("timestamp")),
        "flow_code":       flow.get("code"),
        "trigger_type":    flow.get("trigger_type"),
        "endpoint_name":   error.get("endpoint_name"),
        "error_code":      error.get("code"),
        "error_summary":   error.get("summary"),
        "semantic_text":   semantic_text,
        "raw_json":        json.dumps(raw_log),
        "normalized_json": json.dumps(normalized_log),
        "vector":          _to_vector_array(embedding),
    }


# ── INSERT ─────────────────────────────────────────────────────────────────────

def insert_log(
    normalized_log: dict,
    raw_log: list,
    embedding: list[float],
    semantic_text: str,
    jira_id: str | None = None
) -> str:
    """
    Inserts a log record into OLL_LOGS.

    Args:
        normalized_log:  Normalized log dict (output of normalize_log).
        raw_log:         Original raw log list (JSON array).
        embedding:       Vector embedding (output of generate_embedding).
        semantic_text:   Text used to generate the embedding.
        jira_id:         Associated Jira issue ID (optional).

    Returns:
        LOG_ID of the inserted record.
    """
    record = _build_record(normalized_log, raw_log, embedding, semantic_text, jira_id)

    logger.info(f"Inserting log into OLL_LOGS | flow_code: {record['flow_code']} | jira_id: {jira_id}")

    conn   = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(INSERT_LOG_SQL, record)
        conn.commit()
        logger.info(f"Insert successful | LOG_ID: {record['log_id']}")
        return record["log_id"]

    except Exception as e:
        conn.rollback()
        logger.error(f"Insert failed: {e}")
        raise

    finally:
        cursor.close()
        conn.close()