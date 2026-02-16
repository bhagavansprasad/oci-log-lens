import json
import sys
from normalizer import normalize_log_from_file
from embedder import generate_embedding
import time

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


def test_normalization(file_path: str) -> dict | None:
    """
    Normalizes a single log file and prints the result.

    Args:
        file_path: Path to the log file.

    Returns:
        Normalized log dict if successful, None if an error occurred.
    """
    print(f"\n{'='*60}")
    print(f"FILE : {file_path}")
    print(f"{'='*60}")
    try:
        result = normalize_log_from_file(file_path)
        print(json.dumps(result, sort_keys=True, indent=4))
        return result
    except Exception as e:
        print(f"[NORMALIZATION ERROR] {e}")
        return None


def test_embedding(normalized_log: dict, file_path: str) -> bool:
    """
    Generates an embedding for a normalized log and prints a summary.

    Args:
        normalized_log: Normalized log dict (output of normalize_log).
        file_path:      Source file path — used for display only.

    Returns:
        True if successful, False if an error occurred.
    """
    print(f"\n--- Embedding: {file_path} ---")
    try:
        embedding = generate_embedding(normalized_log)
        print(f"Dimensions : {len(embedding)}")
        print(f"Sample     : {embedding[:5]}")
        return True
    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return False


def test_all_files(log_files: list[str]) -> tuple[int, int]:
    """
    Runs normalization and embedding tests on all provided log files.

    Args:
        log_files: List of log file paths to test.

    Returns:
        Tuple of (passed count, failed count).
    """
    passed = 0
    failed = 0

    for file_path in log_files:
        normalized = test_normalization(file_path)

        if normalized is None:
            failed += 1
            continue

        time.sleep(1)
        embedding_ok = test_embedding(normalized, file_path)

        if embedding_ok:
            passed += 1
        else:
            failed += 1
        time.sleep(5)

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
    passed, failed = test_all_files(LOG_FILES)
    print_summary(passed, failed, len(LOG_FILES))
    sys.exit(0 if failed == 0 else 1)