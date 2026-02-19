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
import hashlib
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
    LOG_HASH,
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
    :log_hash,
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

    raw_json_str = json.dumps(raw_log, sort_keys=True)
    log_hash     = hashlib.sha256(raw_json_str.encode()).hexdigest()

    return {
        "log_id":          str(uuid.uuid4()),
        "log_hash":        log_hash,
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



# ── DUPLICATE CHECK SQL ────────────────────────────────────────────────────────

CHECK_DUPLICATE_SQL = """
SELECT COUNT(*) FROM OLL_LOGS WHERE LOG_HASH = :log_hash
"""


# ── DUPLICATE CHECK ────────────────────────────────────────────────────────────

def check_duplicate(log_hash: str) -> bool:
    """
    Check if a log with the given hash already exists in OLL_LOGS.
    
    Args:
        log_hash: SHA256 hash of the raw log
        
    Returns:
        True if duplicate exists, False otherwise
    """
    conn   = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(CHECK_DUPLICATE_SQL, {"log_hash": log_hash})
        count = cursor.fetchone()[0]
        return count > 0
    
    except Exception as e:
        logger.error(f"Duplicate check failed: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()

# ── SEARCH SQL ─────────────────────────────────────────────────────────────────

SEARCH_SIMILAR_SQL = """
SELECT
    LOG_ID,
    JIRA_ID,
    FLOW_CODE,
    TRIGGER_TYPE,
    ERROR_CODE,
    ERROR_SUMMARY,
    VECTOR_DISTANCE(VECTOR, :query_vector, COSINE) AS SIMILARITY_SCORE
FROM
    OLL_LOGS
ORDER BY
    VECTOR_DISTANCE(VECTOR, :query_vector, COSINE)
FETCH FIRST :top_n ROWS ONLY
"""


# ── SEARCH ─────────────────────────────────────────────────────────────────────

def search_similar_logs(
    query_embedding: list[float],
    top_n: int = 5
) -> list[dict]:
    """
    Searches OLL_LOGS for the most similar logs using vector cosine similarity.

    Args:
        query_embedding: Vector embedding of the query log (output of generate_embedding).
        top_n:           Number of top similar results to return (default: 5).

    Returns:
        List of dicts, each containing:
        - log_id
        - jira_id
        - flow_code
        - trigger_type
        - error_code
        - error_summary
        - similarity_score  (0.0 = identical, 1.0 = completely different — cosine distance)
    """
    query_vector = _to_vector_array(query_embedding)

    logger.info(f"Searching OLL_LOGS for Top-{top_n} similar logs ...")

    conn   = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(SEARCH_SIMILAR_SQL, {
            "query_vector": query_vector,
            "top_n":        top_n
        })

        columns = [col[0].lower() for col in cursor.description]
        rows    = cursor.fetchall()

        # Read all LOB/CLOB fields while connection is still open
        results = []
        for row in rows:
            record = {}
            for col, val in zip(columns, row):
                if hasattr(val, "read"):
                    record[col] = val.read()
                else:
                    record[col] = val
            results.append(record)

        logger.info(f"Search complete. {len(results)} results returned.")
        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise

    finally:
        cursor.close()
        conn.close()