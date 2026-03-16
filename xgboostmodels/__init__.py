"""
Per-attribute XGBoost models and aggregation helpers.

This package trains one XGBoost classifier per attribute (name, phone, web,
address, category) using the shared feature engineering in `scripts.features`
and label logic in `scripts.labels`. A rule-based aggregation step then
combines the per-attribute winners into record-level 4-class / 3-class /
binary labels, mirroring the existing labeling conventions.
"""

