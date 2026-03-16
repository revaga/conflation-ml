"""
Run all pipelines needed for unified metrics so that output parquets exist.
Skips a pipeline if its output already exists and is newer than golden (unless --force).
Run from repo root: python scripts/run_all_for_metrics.py [--force]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"
GOLDEN_200 = DATA_DIR / "golden_dataset_200.parquet"

PIPELINES = [
    ("create_synthetic_4class_golden", "scripts/create_synthetic_4class_golden.py", DATA_DIR / "synthetic_4class_golden.parquet"),
    ("xgboost_binary_alt_base", "scripts/xgboost_binary_alt_base.py", DATA_DIR / "xgboost_binary_results.parquet"),
    ("xgboostbinary (3-class)", "scripts/xgboostbinary.py", DATA_DIR / "xgboost_results.parquet"),
    ("xgboost_multiclass", "scripts/xgboost_multiclass.py", DATA_DIR / "xgboost_multiclass_results.parquet"),
    ("randomforest_binary_alt_base", "scripts/randomforest_binary_alt_base.py", DATA_DIR / "randomforest_binary_results.parquet"),
    ("rule_based_logic", "external_validation/rule_based_logic.py", DATA_DIR / "rule_based.parquet"),
    ("fetch_truth_google", "external_validation/fetch_truth_google.py", DATA_DIR / "ground_truth_google_golden.parquet"),
    ("fetch_truth_scrape", "external_validation/fetch_truth_scrape.py", DATA_DIR / "ground_truth_scrape_golden.parquet"),
    ("phase5_full_pipeline", "scripts/phase5_full_pipeline.py", DATA_DIR / "phase5_full_results.parquet"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipelines for unified metrics if outputs missing or stale")
    parser.add_argument("--force", action="store_true", help="Run all pipelines even if output exists")
    args = parser.parse_args()

    golden_mtime = GOLDEN_200.stat().st_mtime if GOLDEN_200.exists() else 0

    for name, script_rel, output_path in PIPELINES:
        script_path = _REPO_ROOT / script_rel
        if not script_path.exists():
            print(f"Skip {name}: script not found {script_path}", file=sys.stderr)
            continue
        if not args.force and output_path.exists():
            out_mtime = output_path.stat().st_mtime
            if out_mtime >= golden_mtime:
                print(f"Skip {name}: output exists and is up to date: {output_path.name}")
                continue
        print(f"Run {name} ...")
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(_REPO_ROOT),
                check=True,
                timeout=3600,
            )
            print(f"  -> {output_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"  Script not found: {script_path}", file=sys.stderr)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    print("Done. Run: python scripts/unified_metrics_golden200.py --report")


if __name__ == "__main__":
    main()
