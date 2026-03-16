## Places Attribute Conflation (ML)

Creating a single reliable record from multiple location sources using a mix of **rule-based logic**, **tree-based ML models**, and **small/large language models (SLMs/LLMs)**.

This repository builds and evaluates attribute-level conflation models over pre-matched place pairs, producing a hand-labeled golden set (200 rows), synthetic extensions, and unified metrics comparing many methods side by side.

---

### Problem

Real‑world places often appear in multiple datasets with inconsistent, outdated, or conflicting attributes (phone, website, address, category, etc.).  
Given a **base** record and an **alternate/conflated** record for the same place, we want to decide, per attribute, which side is more trustworthy and what the final combined record should be.

The project focuses on:

- **Per‑attribute winners**: `base`, `alt`, `both`, `none` (and `real` in external validation).
- **Record‑level labels**: 2‑class (base vs alt), 3‑class (base / both / alt), and 4‑class (none / alt / base / both).
- **Unified evaluation**: comparing XGBoost, Random Forest, SLMs, and rule-based / external pipelines on a consistent golden dataset.

---

### Data Overview

All core datasets live under `data/`:

- **Golden / labels**
  - `golden_dataset_200.parquet`: primary hand‑labeled golden set (attribute winners + record‑level labels).
  - `golden_dataset_100_aligned.parquet`: aligned subset used for Golden‑100 metrics.
  - `phase3_slm_labeled*.parquet`: SLM‑labeled per‑attribute winners and 4‑class labels (Gemma3, Kimi, GPT‑4o mini).
- **Synthetic**
  - `synthetic_4class_golden.parquet`: synthetic 4‑class golden labels derived from golden 200.
  - `synthetic_4class_kimi.parquet`: synthetic labels created with Kimi (`create_synthetic_4class_*`).
- **Model outputs**
  - `xgboost_results.parquet`, `xgboost_multiclass_results.parquet`, `xgboost_binary_results.parquet`
  - `xgboost_per_attr_results.parquet`, `xgboost_per_attr_ensemble_results.parquet`
  - `randomforest_binary_results.parquet`
- **External / rule‑based truth**
  - `ground_truth_google_golden.parquet`, `ground_truth_scrape_golden.parquet`
  - `ground_truth_google_no_fallback.parquet`, `ground_truth_scrape_no_fallback.parquet`
  - `ground_truth_rule_based.parquet`, `rule_based_100.parquet`

Each row is a **pre‑matched pair** of records. Columns without a prefix refer to the conflated/alternate record; `base_`‑prefixed columns refer to the base record (e.g., `phones` vs `base_phones`, `addresses` vs `base_addresses`).

---

### Repository Layout

```
conflation-ml/
├── data/                      # Golden datasets, synthetic labels, model outputs, external truth
├── scripts/
│   ├── phase1_data_prep.py    # Initial cleaning and normalization
│   ├── phase2_similarity.py   # Similarity features over base vs alt attributes
│   ├── features.py            # Single source of truth for feature engineering
│   ├── labels.py              # Label schemes, mapping between 2/3/4‑class, attr_*_winner logic
│   ├── schema.py              # Data schema and validation helpers
│   ├── parquet_io.py          # Safe Parquet reading / writing
│   ├── normalization.py       # Phone / web / address normalization utilities
│   ├── phonenumber_validator.py
│   ├── website_validator.py
│   ├── validator_cache.py     # DiskCache helpers for external calls
│   ├── golden_dataset_maker.py
│   ├── slm_attribute_labeler.py
│   ├── finetune_gemma3_attr_lora.py
│   ├── create_synthetic_4class_golden.py
│   ├── create_synthetic_4class_kimi.py
│   ├── generate_negatives.py
│   ├── xgboostbinary.py
│   ├── xgboost_multiclass.py
│   ├── xgboost_binary_alt_base.py
│   ├── randomforest_binary_alt_base.py
│   ├── phase4_eval.py
│   ├── phase5_full_pipeline.py
│   ├── pipeline_eval.py
│   ├── final_eval.py
│   ├── run_phase3_ollama_and_hf.py
│   ├── run_all_for_metrics.py # Orchestrates all pipelines needed for unified metrics
│   ├── unified_metrics_golden200.py
│   ├── inspect_parquet.py
│   ├── compare_parquet_datasets.py
│   ├── diff_parquet.py
│   ├── detect_data_drift.py
│   ├── show_golden_columns.py
│   └── api_conflator.py
├── external_validation/
│   ├── compare.py
│   └── README.md              # Detailed docs for Google / scrape / rule-based truth pipelines
├── reports/
│   ├── metrics_summary_golden200.md
│   └── metrics_summary_golden100.md
├── scripts/_archive/DEPRECATED.md
├── requirements.txt
└── README.md
```

---

## Setup

- **Python**: 3.10+ recommended.
- **Install dependencies** (from repo root):

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows
# or: source .venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
```

Some pipelines require API keys (see below for external validation and SLM labeling).

---

## Core Concepts

- **Attributes**: phone, web, address, category (and others for internal features).
- **Per‑attribute winner** (`attr_*_winner`): which side is correct (`base`, `alt`, `both`, `none`, or `real` in external truth).
- **Record‑level labels**:
  - **2‑class**: `base` vs `alt` (alt = “conflated is better or at least as good”).
  - **3‑class**: `base` / `both` / `alt`.
  - **4‑class**: `none` / `alt` / `base` / `both`.
- **Methods compared**:
  - Rule‑based (no external data).
  - Google Maps / Places and web scraping pipelines.
  - XGBoost and Random Forest baselines.
  - SLM / LLM attribute labelers (Gemma3 4B, Kimi, GPT‑4o mini) and a small fine‑tuned model.

The logic for label mapping and recomputing record‑level labels from per‑attribute winners lives in `scripts/labels.py`.

---

## Running the Main Pipelines

### 1. Run all pipelines needed for metrics

From the repo root:

```bash
python scripts/run_all_for_metrics.py
# or to force re‑run everything, ignoring existing outputs:
python scripts/run_all_for_metrics.py --force
```

This script:

- Ensures `data/golden_dataset_200.parquet` exists.
- Runs synthetic generation, all XGBoost / Random Forest baselines, rule‑based logic, Google, scrape, and `phase5_full_pipeline.py` as needed.
- Skips any pipeline where the output Parquet is newer than the golden (unless `--force` is given).

### 2. Compute unified metrics and write reports

```bash
python scripts/unified_metrics_golden200.py --report
```

This:

- Loads `golden_dataset_200.parquet` and `synthetic_4class_golden.parquet` (if present).
- Computes:
  - 3‑class metrics vs golden 200.
  - 4‑class metrics vs synthetic 4‑class golden.
  - Binary metrics vs 2‑class labels.
  - Per‑attribute winner agreement and accuracy.
  - Value similarity vs golden using external truth parquets.
- Writes:
  - CSVs under `reports/` (e.g., `unified_metrics_3class.csv`, `unified_metrics_4class.csv`, `unified_metrics_binary.csv`).
  - A human‑readable summary markdown: `reports/metrics_summary_golden200.md`.

To run on a subset (e.g., Golden‑100):

```bash
python scripts/unified_metrics_golden200.py --golden-limit 100 --report
```

This writes suffixed outputs such as `metrics_summary_golden100.md`.

---

## External Validation Pipelines

External validation uses **real‑world data** (Google Places API, web scraping + search, or pure rule‑based logic) to produce `truth_*_winner` and `truth_*_value` columns per attribute.

- Documentation and commands live in `external_validation/README.md`.
- Typical usage (from repo root):

```bash
python external_validation/fetch_truth_google.py --limit 200
python external_validation/fetch_truth_scrape.py --limit 200
python external_validation/rule_based_logic.py
```

These produce:

- `data/ground_truth_google_golden.parquet`
- `data/ground_truth_scrape_golden.parquet`
- `data/ground_truth_rule_based.parquet`

They are then consumed by `scripts/unified_metrics_golden200.py` for similarity and per‑attribute agreement analyses.

**Env / config**:

- Google Places requires `GOOGLE_PLACES_API_KEY` in your environment or in `api_keys.env` at the repo root.
- See `external_validation/README.md` for details and caveats about scraping/search ToS.

---

## SLM / LLM Labeling

Per‑attribute winners and record‑level labels can be generated using SLMs / LLMs:

- `scripts/slm_attribute_labeler.py`: prompts a model (local or hosted) to label each row.
- `scripts/run_phase3_ollama_and_hf.py`: helper for running multiple SLM configurations and saving labeled Parquets.
- `scripts/finetune_gemma3_attr_lora.py`: example of fine‑tuning Gemma3 for this attribute‑labeling task.

The outputs are `phase3_slm_labeled*.parquet` files in `data/`, which are picked up automatically by `unified_metrics_golden200.py`.

---

## Inspecting and Debugging Data

Helpful utility scripts (run from repo root):

- `python scripts/inspect_parquet.py --path data/golden_dataset_200.parquet`
- `python scripts/compare_parquet_datasets.py --a data/xgboost_results.parquet --b data/xgboost_multiclass_results.parquet`
- `python scripts/diff_parquet.py --a ... --b ...`
- `python scripts/detect_data_drift.py`
- `python scripts/show_golden_columns.py`

These are designed for quick sanity checks while iterating on features, labels, and pipelines.

---

## Schema Reference

The project is motivated by and aligned with the Overture Places schema. For field types, structure, and definitions, see:

- **Overture Places schema**: `https://docs.overturemaps.org/schema/reference/places/place/`

Columns in the working Parquet files are adapted for ML (e.g., normalized phone / web / address fields, similarity scores, and derived labels), but the underlying semantics follow the Overture definitions.

---

## Repro Checklist

1. **Create and activate a virtual environment**, then `pip install -r requirements.txt`.
2. Ensure `data/golden_dataset_200.parquet` exists (or generate / copy it into `data/`).
3. (Optional) Configure external validation (`GOOGLE_PLACES_API_KEY`, scraping environment).
4. Run `python scripts/run_all_for_metrics.py` to generate all model and truth outputs.
5. Run `python scripts/unified_metrics_golden200.py --report` to produce metrics CSVs and markdown reports in `reports/`.
6. Inspect `reports/metrics_summary_golden200.md` (and `metrics_summary_golden100.md` if using a subset) for headline results across all methods.

