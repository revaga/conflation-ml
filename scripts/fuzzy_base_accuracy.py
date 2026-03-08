"""
Base accuracy: using only the fuzzy total_similarity from phase2_similarity,
classify as 1 (accept match) or 0 (keep base) at threshold 0.671 and report
accuracy vs golden_label. Uses data/fuzzy_scored.parquet (run fuzzy_match_score.py
or phase2_similarity.py first) or data/phase2_scored.parquet.

Run from project root: python scripts/fuzzy_base_accuracy.py
"""
import sys
from pathlib import Path

# Allow importing phase2_similarity when run as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_similarity import (
    PHASE2_SCORED_PATH,
    PHASE3_PATH,
    report_base_accuracy,
    BASE_ACCURACY_THRESHOLD,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUZZY_SCORED_PATH = PROJECT_ROOT / "data" / "fuzzy_scored.parquet"


def main():
    # Prefer fuzzy_scored (has golden_label if built from phase3), else phase2_scored
    if FUZZY_SCORED_PATH.exists():
        scored_path = FUZZY_SCORED_PATH
    elif PHASE2_SCORED_PATH.exists():
        scored_path = PHASE2_SCORED_PATH
    else:
        print("No scored parquet found. Running fuzzy_match_score.py...")
        import subprocess
        rc = subprocess.call(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "fuzzy_match_score.py")],
            cwd=str(PROJECT_ROOT),
        )
        if rc != 0 or not FUZZY_SCORED_PATH.exists():
            print("Run first: python scripts/fuzzy_match_score.py or scripts/phase2_similarity.py")
            return 1
        scored_path = FUZZY_SCORED_PATH

    report_base_accuracy(scored_path, golden_path=PHASE3_PATH, threshold=BASE_ACCURACY_THRESHOLD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
