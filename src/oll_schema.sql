-- ============================================================
-- OIC-LogLens — Oracle 26ai Schema
-- Prefix  : OLL (OIC Log Lens)
-- User    : EA_APP | DB: FREEPDB1
-- Version : 1.0
-- ============================================================

-- ── Drop existing objects safely ─────────────────────────────

BEGIN
    EXECUTE IMMEDIATE 'DROP INDEX OLL_LOGS_VIDX';
EXCEPTION
    WHEN OTHERS THEN NULL;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE OLL_LOGS';
EXCEPTION
    WHEN OTHERS THEN NULL;
END;
/

-- ── Main Table ───────────────────────────────────────────────

CREATE TABLE OLL_LOGS (
    LOG_ID            VARCHAR2(100)    PRIMARY KEY,
    LOG_HASH          VARCHAR2(64)     NOT NULL,
    JIRA_ID           VARCHAR2(100),
    LOG_TYPE          VARCHAR2(20),
    EVENT_TIME        TIMESTAMP,
    FLOW_CODE         VARCHAR2(200),
    TRIGGER_TYPE      VARCHAR2(50),
    ENDPOINT_NAME     VARCHAR2(200),
    ERROR_CODE        VARCHAR2(100),
    ERROR_SUMMARY     CLOB,
    SEMANTIC_TEXT     CLOB,
    RAW_JSON          CLOB,
    NORMALIZED_JSON   CLOB,
    VECTOR            VECTOR(3072, FLOAT32)
);

ALTER TABLE OLL_LOGS ADD CONSTRAINT OLL_LOGS_HASH_UQ UNIQUE (LOG_HASH);

-- ── Column Comments ───────────────────────────────────────────

COMMENT ON TABLE OLL_LOGS IS
    'OIC-LogLens: Stores ingested OIC flow logs with normalized metadata, raw content, and vector embeddings for semantic deduplication.';

COMMENT ON COLUMN OLL_LOGS.LOG_ID IS
    'Primary key. UUID generated at insert time. Uniquely identifies each ingested log record.';

-- Add with other comments:
COMMENT ON COLUMN OLL_LOGS.LOG_HASH IS
    'SHA256 hash of the raw log JSON. Used to prevent duplicate log ingestion. Same raw log will always produce the same hash.';

COMMENT ON COLUMN OLL_LOGS.JIRA_ID IS
    'Associated Jira issue ID for this log. Used as the deduplication reference returned during similarity search.';

COMMENT ON COLUMN OLL_LOGS.LOG_TYPE IS
    'Classification of the log. Values: error | informational. Populated from normalized_json.log_type.';

COMMENT ON COLUMN OLL_LOGS.EVENT_TIME IS
    'Timestamp of the flow execution event. Populated from normalized_json.flow.timestamp (converted from epoch ms).';

COMMENT ON COLUMN OLL_LOGS.FLOW_CODE IS
    'OIC integration flow code that generated this log. Populated from normalized_json.flow.code.';

COMMENT ON COLUMN OLL_LOGS.TRIGGER_TYPE IS
    'How the flow was triggered. Values: rest | soap | scheduled. Populated from normalized_json.flow.trigger_type.';

COMMENT ON COLUMN OLL_LOGS.ENDPOINT_NAME IS
    'Name of the endpoint where the error occurred. Populated from normalized_json.error.endpoint_name.';

COMMENT ON COLUMN OLL_LOGS.ERROR_CODE IS
    'Short error code from the log. Examples: Execution failed, 401, 404, 503. Populated from normalized_json.error.code.';

COMMENT ON COLUMN OLL_LOGS.ERROR_SUMMARY IS
    'One-line human readable error summary. Populated from normalized_json.error.summary.';

COMMENT ON COLUMN OLL_LOGS.SEMANTIC_TEXT IS
    'Concatenated text string built from key normalized fields. This is the input used to generate the vector embedding.';

COMMENT ON COLUMN OLL_LOGS.RAW_JSON IS
    'Original raw OIC log file content as-is. Stored for traceability and reprocessing.';

COMMENT ON COLUMN OLL_LOGS.NORMALIZED_JSON IS
    'Full normalized log JSON produced by the LLM normalization pipeline. Stored for audit and future use.';

COMMENT ON COLUMN OLL_LOGS.VECTOR IS
    'Vector embedding (3072 dimensions, FLOAT32) generated from SEMANTIC_TEXT using Gemini text-embedding-004. Used for cosine similarity search.';

-- ── Vector Index (HNSW Cosine) ───────────────────────────────
CREATE VECTOR INDEX OLL_LOGS_VIDX
ON OLL_LOGS(VECTOR)
ORGANIZATION INMEMORY GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- ── Metadata Indexes ─────────────────────────────────────────

CREATE INDEX OLL_LOGS_FLOW_IDX  ON OLL_LOGS(FLOW_CODE);
CREATE INDEX OLL_LOGS_JIRA_IDX  ON OLL_LOGS(JIRA_ID);
CREATE INDEX OLL_LOGS_TIME_IDX  ON OLL_LOGS(EVENT_TIME);
CREATE INDEX OLL_LOGS_TYPE_IDX  ON OLL_LOGS(LOG_TYPE);

-- ── Verify ───────────────────────────────────────────────────

SELECT 'TABLE CREATED OK' AS STATUS
FROM USER_TABLES
WHERE TABLE_NAME = 'OLL_LOGS';

SELECT INDEX_NAME, INDEX_TYPE
FROM USER_INDEXES
WHERE TABLE_NAME = 'OLL_LOGS';

SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH
FROM USER_TAB_COLUMNS
WHERE TABLE_NAME = 'OLL_LOGS'
ORDER BY COLUMN_ID;