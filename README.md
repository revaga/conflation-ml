# Places Attribute Conflation

**Project A · Winter 2026 · CRWN 102**

Creating a single reliable record from multiple location sources.

---

## Overview

Real-world places often appear in multiple datasets with inconsistent, outdated, or conflicting information. This project tackles the problem of **attribute-level conflation**: given multiple representations of the same place, how do we decide which attributes (phone, website, email, etc.) are the most accurate?

Our goal is to produce a high-quality golden dataset and evaluate different strategies—rule-based logic vs. machine learning—for selecting the best attributes.

This project is developed as part of coursework at the University of California, Santa Cruz, in partnership with the [Overture Maps Foundation](https://overturemaps.org/), and is motivated by the structure and constraints of the Overture Maps Places dataset.

---

### Data Context

This repository works with **pre-matched pairs** of place records. Each row represents a conflation: one place (the *base*) merged with attributes from other sources to produce a conflated record. We use this data to understand and evaluate how well attributes from different datasets can be combined into a single, trustworthy place entry.

### Team

**Neha Ashwin, Reva Agarwal**

---

## Project Structure

```
neha-reva-places-attribute-conflation/
├── data/
│   ├── project_a_samples.parquet       # Raw sample data (~2,000 pairs)
│   ├── phase1_processed.parquet        # Normalized & basic similarities
│   ├── phase3_slm_labeled.parquet      # SLM-labeled attribute winners
│   └── golden_dataset_200.parquet      # Hand-labeled truth set
├── scripts/
│   ├── features.py                     # Single Source of Truth for feature engineering
│   ├── labels.py                       # Centralized label maps & derivation logic
│   ├── schema.py                       # Data validation contracts
│   ├── parquet_io.py                   # Safe Parquet reading utility
│   ├── validator_cache.py              # DiskCache for external API calls
│   ├── slm_attribute_labeler.py        # LLM-based labeling script
│   ├── xgboostbinary.py                # 2-stage XGBoost Classifier
│   ├── xgboost_multiclass.py           # Multiclass XGBoost Classifier
│   └── phase5_full_pipeline.py         # End-to-end evaluation pipeline
├── tests/
│   └── test_feature_parity.py          # Verification script for feature parity
├── README.md
└── requirements.txt
```

---

## Exploring the Data

From the project root:

```bash
source overture/bin/activate
python scripts/project_data.py
```

This prints a dataset overview including:

- **Schema** — All columns and types
- **Row count** — Total records
- **Null counts** — Which attributes are often missing
- **Confidence distribution** — Conflated vs base record confidence
- **Sample rows** — Example key attributes
- **Uniqueness** — `id` and `base_id` cardinality

---

## Data Schema

Each row is a pre-matched pair. Columns without a prefix come from the **conflated** record; columns with the `base_` prefix come from the **base** (original) place record.

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | Conflated record ID |
| `base_id` | VARCHAR | Base place record ID |
| `sources` | VARCHAR | JSON array of contributing sources (e.g., meta, msft) |
| `names` | VARCHAR | Conflated names (JSON: `primary`, `alternate`) |
| `base_names` | VARCHAR | Base names |
| `categories` | VARCHAR | Conflated categories (e.g., `shipping_center`, `post_office`) |
| `base_categories` | VARCHAR | Base categories |
| `confidence` | DOUBLE | Conflation confidence score |
| `base_confidence` | DOUBLE | Base record confidence |
| `websites` | VARCHAR | Website URLs |
| `base_websites` | VARCHAR | Base websites |
| `socials` | VARCHAR | Social media links |
| `base_socials` | VARCHAR | Base socials |
| `emails` | INTEGER | Email count (often sparse) |
| `base_emails` | VARCHAR | Base emails |
| `phones` | VARCHAR | Phone numbers |
| `base_phones` | VARCHAR | Base phones |
| `brand` | VARCHAR | Brand info |
| `base_brand` | VARCHAR | Base brand |
| `addresses` | VARCHAR | Conflated address (JSON: freeform, locality, region, etc.) |
| `base_addresses` | VARCHAR | Base addresses |
| `base_sources` | VARCHAR | Base source metadata |

### Key Concepts

- **Base record** — The original place from one dataset (e.g., Microsoft); has `base_*` columns.
- **Conflated record** — The merged result, combining attributes from multiple sources; non-prefixed columns.
- **Confidence** — Indicates the confidence that a place exists. Base confidence is typically ~0.77.

---

## Phase 1: Preprocessing

## Phase 2: Feature Engineering

## Phase 3: Labeling

## Phase 4: Model Training

## Phase 5: Evaluation

## XGBoost
Phone number format validation with phonenumber python library and country codes
Website existence validation with https GET request


## Schema Reference

Overture Places schema (field types, structure, and definitions):

**[Overture Places Schema](https://docs.overturemaps.org/schema/reference/places/place/)**
