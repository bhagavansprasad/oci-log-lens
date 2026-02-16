import json
import sys
from normalizer import normalize_log_from_file

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


def test_single_file(file_path: str) -> bool:
    """
    Normalizes a single log file and prints the result.

    Args:
        file_path: Path to the log file.

    Returns:
        True if successful, False if an error occurred.
    """
    print(f"\n{'='*60}")
    print(f"FILE : {file_path}")
    print(f"{'='*60}")
    try:
        result = normalize_log_from_file(file_path)
        print(json.dumps(result, sort_keys=True, indent=4))
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_all_files(log_files: list[str]) -> tuple[int, int]:
    """
    Runs normalization test on all provided log files.

    Args:
        log_files: List of log file paths to test.

    Returns:
        Tuple of (passed count, failed count).
    """
    passed = 0
    failed = 0

    for file_path in log_files:
        if test_single_file(file_path):
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
    passed, failed = test_all_files(LOG_FILES)
    print_summary(passed, failed, len(LOG_FILES))
    sys.exit(0 if failed == 0 else 1)
    