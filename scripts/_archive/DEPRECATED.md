# Deprecated Scripts

The following scripts have been moved to this archive to reduce cognitive load and clarify the primary data pipeline. They are either experimental prototypes, one-time fix-ups, or superseded by more robust tools.

| Script | Reason for Deprecation | Superseded By |
|--------|------------------------|----------------|
| `_analysis.py` | One-off feature analysis with non-standard path hacks. | `features.py` / `tests/` |
| `_strategy_test.py` | Experimental GBC implementation comparison. | `xgboost_multiclass.py` |
| `_hyperparam_search.py` | Early hyperparameter search prototype. | `xgboost_multiclass.py` (built-in GridSearchCV) |
| `check_golden_200.py` | Minor 14-line inspection utility. | `data_utils.py` (planned) |
| `inspect_golden.py` | Redundant inspection; used hardcoded absolute paths. | `data_utils.py` (planned) |
| `print_golden_dataset.py` | Simple CLI printer. | Standard dataframe `head()` in notebooks/scripts. |
| `show_parquet_sample.py` | Generic parquet viewer. | `inspect_parquet.py` |
| `rewrite_golden_parquet.py` | One-time PyArrow/FastParquet compatibility fix. | `parquet_io.py` |
| `fix_and_apply_3class.py` | One-time label correction; logic duplicated elsewhere. | `labels.py` / `derive_labels.py` |
| `standardize_labels.py` | Contained hardcoded absolute Windows paths; brittle. | `data_utils.py` (planned) |
| `phase2_prep_for_llm.py` | Early SLM prototype. | `slm_attribute_labeler.py` |
| `phase3_slm.py` | Broken schema handling (`same_place` vs `label`). | `slm_attribute_labeler.py` |
| `address_validator.py` | Orphaned script; zero imports in the codebase. | `features.py` (planned) |
| `label_data.py` | Superseded by interactive `golden_dataset_maker.py`. | `golden_dataset_maker.py` |
| `apply_3class_to_golden_200.py` | Logic for deriving 3-class from winners is now in `labels.py`. | `labels.py` |
| `apply_phone_logic_to_golden.py` | Phone validation logic centralized. | `validators.py` |
| `golden_dataset_maker_200.py` | Duplicate of 100-record version with different offset. | `golden_dataset_maker.py` |
| `inspect_processed.py` | Redundant with `inspect_parquet.py`. | `inspect_parquet.py` |
