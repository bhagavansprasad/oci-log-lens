import json
import sys
from normalizer import normalize_log_from_file
from embedder import generate_embedding
from prompts import get_embedding_text
from db import insert_log

LOG_FILES = [
    "flow-logs/01_flow-log.json",
    "flow-logs/02_flow-log.json",
    "flow-logs/03_flow-log.json",
    "flow-logs/04_flow-log.json",
    "flow-logs/05_flow-log.json",
    "flow-logs/06_flow-log.json",
    "flow-logs/07_flow-log.json",
    "flow-logs/08_flow-log.json",
]

# Placeholder Jira IDs for testing — one per log file
JIRA_IDS = [
    "OIC-1001",
    "OIC-1002",
    "OIC-1003",
    "OIC-1004",
    "OIC-1005",
    "OIC-1006",
    "OIC-1007",
    "OIC-1008",
]


def test_normalization(file_path: str) -> tuple[dict, list] | tuple[None, None]:
    """
    Normalizes a single log file and prints the result.

    Args:
        file_path: Path to the log file.

    Returns:
        Tuple of (normalized_log dict, raw_log list) if successful.
        Tuple of (None, None) if an error occurred.
    """
    print(f"\n{'='*60}")
    print(f"FILE : {file_path}")
    print(f"{'='*60}")
    try:
        with open(file_path) as f:
            raw_log = json.load(f)
        normalized = normalize_log_from_file(file_path)
        print(json.dumps(normalized, sort_keys=True, indent=4))
        return normalized, raw_log
    except Exception as e:
        print(f"[NORMALIZATION ERROR] {e}")
        return None, None


def test_embedding(normalized_log: dict, file_path: str) -> tuple[list, str] | tuple[None, None]:
    """
    Generates an embedding for a normalized log and prints a summary.

    Args:
        normalized_log: Normalized log dict (output of normalize_log).
        file_path:      Source file path — used for display only.

    Returns:
        Tuple of (embedding list, semantic_text str) if successful.
        Tuple of (None, None) if an error occurred.
    """
    print(f"\n--- Embedding: {file_path} ---")
    try:
        semantic_text = get_embedding_text(normalized_log)
        embedding = generate_embedding(normalized_log)
        print(f"Dimensions : {len(embedding)}")
        print(f"Sample     : {embedding[:5]}")
        return embedding, semantic_text
    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return None, None


def test_db_insert(
    normalized_log: dict,
    raw_log: list,
    embedding: list,
    semantic_text: str,
    jira_id: str,
    file_path: str
) -> bool:
    """
    Inserts a log record into Oracle 26ai OLL_LOGS table.

    Args:
        normalized_log: Normalized log dict.
        raw_log:        Original raw log list.
        embedding:      Vector embedding.
        semantic_text:  Text used to generate the embedding.
        jira_id:        Associated Jira issue ID.
        file_path:      Source file path — used for display only.

    Returns:
        True if successful, False if an error occurred.
    """
    print(f"\n--- DB Insert: {file_path} ---")
    try:
        log_id = insert_log(
            normalized_log=normalized_log,
            raw_log=raw_log,
            embedding=embedding,
            semantic_text=semantic_text,
            jira_id=jira_id
        )
        print(f"LOG_ID : {log_id}")
        return True
    except Exception as e:
        print(f"[DB INSERT ERROR] {e}")
        return False


def test_all_files(log_files: list[str], jira_ids: list[str]) -> tuple[int, int]:
    """
    Runs normalization, embedding, and DB insert tests on all log files.

    Args:
        log_files: List of log file paths to test.
        jira_ids:  List of Jira IDs corresponding to each log file.

    Returns:
        Tuple of (passed count, failed count).
    """
    passed = 0
    failed = 0

    for file_path, jira_id in zip(log_files, jira_ids):

        # Step 1 — Normalize
        normalized, raw_log = test_normalization(file_path)
        if normalized is None:
            failed += 1
            continue

        # Step 2 — Embed
        embedding, semantic_text = test_embedding(normalized, file_path)
        if embedding is None:
            failed += 1
            continue

        # Step 3 — Insert
        ok = test_db_insert(normalized, raw_log, embedding, semantic_text, jira_id, file_path)
        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def print_summary(passed: int, failed: int, total: int) -> None:
    """
    Prints the final test summary.

    Args:
        passed: Number of files that passed.
        failed: Number of files that failed.
        total:  Total number of files tested.
    """
    print(f"\n{'='*60}")
    print(f"SUMMARY  |  Passed: {passed}  |  Failed: {failed}  |  Total: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    passed, failed = test_all_files(LOG_FILES, JIRA_IDS)
    print_summary(passed, failed, len(LOG_FILES))
    sys.exit(0 if failed == 0 else 1)