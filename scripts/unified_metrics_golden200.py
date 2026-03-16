"""
Unified metrics evaluation against golden_dataset_200 (3-class, binary) and
synthetic_4class_golden (4-class). Produces CSVs and runs report generation.

Usage (from repo root):
    python scripts/unified_metrics_golden200.py [--report]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.labels import (
    ATTR_ATTRS,
    fourclass_to_threeclass,
    recalculate_3class_label,
    recalculate_4class_label,
)
from scripts.parquet_io import read_parquet_safe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = _REPO_ROOT / "data"
REPORTS_DIR = _REPO_ROOT / "reports"
GOLDEN_200_PATH = DATA_DIR / "golden_dataset_200.parquet"
SYNTHETIC_4CLASS_PATH = DATA_DIR / "synthetic_4class_golden.parquet"

# truth_*_winner columns (no name in external validation)
TRUTH_WINNER_COLS = [
    "truth_phone_winner",
    "truth_web_winner",
    "truth_address_winner",
    "truth_category_winner",
]
# Attributes used for per-attribute accuracy (must have golden attr_*_winner and method output)
PER_ATTR_ACCURACY_ATTRS = ("phone", "web", "address", "category")
# 3-class eval: exclude "none"; only alt, both, base count
VALID_3CLASS = ("alt", "both", "base")


def _norm_label(v) -> str:
    if v is None or (isinstance(v, float) and (pd.isna(v) or str(v) == "nan")):
        return ""
    return str(v).strip().lower()


def _norm_attr_winner_for_compare(v) -> str:
    """Normalize attribute winner for comparison (real -> alt to match golden)."""
    x = _norm_label(v)
    if x == "real":
        return "alt"
    return x


def truth_to_4class(row: pd.Series) -> str:
    """Derive 4-class from truth_*_winner; treat real as alt; for none use base vs alt counts."""
    counts = {"base": 0, "alt": 0, "both": 0, "none": 0}
    for col in TRUTH_WINNER_COLS:
        if col not in row.index:
            return "both"
        v = _norm_label(row.get(col))
        if v == "real":
            v = "alt"
        if v in counts:
            counts[v] += 1
    n_alt = counts["alt"]
    n_base = counts["base"]
    n_both = counts["both"]
    if n_alt > n_base:
        return "alt"
    if n_base > n_alt:
        return "base"
    if n_both > 0:
        return "both"
    # Would be none: use base vs alt (already counted); tie -> both
    return "both"


def truth_to_3class(row: pd.Series) -> str:
    """3-class from truth_*_winner (none mapped via counts to base/alt/both)."""
    l4 = truth_to_4class(row)
    return fourclass_to_threeclass(l4) if l4 else "base"


def normalize_3class_truth(series: pd.Series) -> pd.Series:
    """Map match -> alt for 3-class golden labels."""
    out = series.astype(str).str.strip().str.lower()
    out = out.replace({"match": "alt"})
    return out


def load_golden_200() -> pd.DataFrame:
    if not GOLDEN_200_PATH.exists():
        raise FileNotFoundError(f"Golden 200 not found: {GOLDEN_200_PATH}")
    df = read_parquet_safe(str(GOLDEN_200_PATH))
    df["id"] = df["id"].astype(str)
    if "3class_testlabels" in df.columns:
        df["_truth_3class"] = normalize_3class_truth(df["3class_testlabels"])
    if "2class_testlabels" in df.columns:
        df["_truth_2class"] = df["2class_testlabels"].astype(str).str.strip().str.lower()
    # Binary from 3-class (alt+both→alt, base→base) for reference-style SLM table
    if "3class_testlabels" in df.columns:
        t = df["3class_testlabels"].astype(str).str.strip().str.lower().replace({"match": "alt"})
        df["_truth_2class_from_3class"] = t.apply(lambda x: "alt" if x in ("alt", "both") else "base")
    return df


def load_synthetic_4class() -> pd.DataFrame | None:
    if not SYNTHETIC_4CLASS_PATH.exists():
        logger.warning("Synthetic 4-class golden not found; 4-class metrics will be skipped.")
        return None
    df = read_parquet_safe(str(SYNTHETIC_4CLASS_PATH))
    df["id"] = df["id"].astype(str)
    if "4class_testlabels" in df.columns:
        df["_truth_4class"] = df["4class_testlabels"].astype(str).str.strip().str.lower()
    return df


def metrics_3class(truth: pd.Series, pred: pd.Series, labels_3=("alt", "both", "base")) -> dict:
    """Accuracy, F1 macro, per-class F1 for 3-class."""
    mask = truth.notna() & pred.notna()
    t = truth.loc[mask].astype(str).str.strip().str.lower()
    p = pred.loc[mask].astype(str).str.strip().str.lower()
    if len(t) == 0:
        return {"accuracy": np.nan, "f1_macro": np.nan, "n": 0, "f1_alt": np.nan, "f1_both": np.nan, "f1_base": np.nan}
    acc = accuracy_score(t, p)
    f1_macro = f1_score(t, p, average="macro", labels=list(labels_3), zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        t, p, labels=list(labels_3), zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "n": len(t),
        "f1_alt": f1_c[0],
        "f1_both": f1_c[1],
        "f1_base": f1_c[2],
    }


def metrics_4class(truth: pd.Series, pred: pd.Series) -> dict:
    """Accuracy, F1 macro, per-class F1 for 4-class (none, alt, base, both)."""
    labels_4 = ("none", "alt", "base", "both")
    mask = truth.notna() & pred.notna()
    t = truth.loc[mask].astype(str).str.strip().str.lower()
    p = pred.loc[mask].astype(str).str.strip().str.lower()
    if len(t) == 0:
        return {"accuracy": np.nan, "f1_macro": np.nan, "n": 0}
    acc = accuracy_score(t, p)
    f1_macro = f1_score(t, p, average="macro", labels=list(labels_4), zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        t, p, labels=list(labels_4), zero_division=0
    )
    out = {"accuracy": acc, "f1_macro": f1_macro, "n": len(t)}
    for i, c in enumerate(labels_4):
        out[f"f1_{c}"] = f1_c[i]
    return out


def metrics_binary(truth: pd.Series, pred: pd.Series) -> dict:
    mask = truth.notna() & pred.notna()
    t = truth.loc[mask].astype(str).str.strip().str.lower()
    p = pred.loc[mask].astype(str).str.strip().str.lower()
    if len(t) == 0:
        return {"accuracy": np.nan, "f1": np.nan, "n": 0}
    acc = accuracy_score(t, p)
    f1 = f1_score(t, p, average="binary", pos_label="alt", zero_division=0)
    return {"accuracy": acc, "f1": f1, "n": len(t)}


def run_evaluations(golden: pd.DataFrame, synthetic: pd.DataFrame | None) -> tuple[list, list, list, list, list, list, dict]:
    results_3class = []
    results_4class = []
    results_binary = []
    similarity_rows = []
    per_attr_rows = []
    per_attr_accuracy_rows: list[dict] = []
    slm_binary_3class_truth: dict = {}

    golden_ids = set(golden["id"])
    synth_ids = set(synthetic["id"]) if synthetic is not None else set()

    def merge_golden(df: pd.DataFrame, pred_col: str | None = None, cols: list | None = None) -> pd.DataFrame:
        df = df.copy()
        df["id"] = df["id"].astype(str)
        merge_cols = ["id"]
        if pred_col and pred_col in df.columns:
            merge_cols.append(pred_col)
        if cols:
            for c in cols:
                if c not in merge_cols and c in df.columns:
                    merge_cols.append(c)
        use = golden.merge(df[merge_cols], on="id", how="inner", suffixes=("", "_pred"))
        return use

    def filter_3class_eval(m: pd.DataFrame) -> pd.DataFrame:
        """Restrict to rows where golden 3-class truth is alt/both/base (exclude none)."""
        if m.empty or "_truth_3class" not in m.columns:
            return m
        t = m["_truth_3class"].astype(str).str.strip().str.lower()
        return m[t.isin(VALID_3CLASS)].copy()

    def merge_synth(df: pd.DataFrame, pred_col: str) -> pd.DataFrame | None:
        if synthetic is None:
            return None
        df = df.copy()
        df["id"] = df["id"].astype(str)
        common = synthetic.merge(df[["id", pred_col]], on="id", how="inner")
        return common

    # ----- Binary: XGBoost binary, Random Forest, per-attribute XGBoost ensemble -----
    if "_truth_2class" in golden.columns:
        for path, pred_col, name in [
            (DATA_DIR / "xgboost_binary_results.parquet", "xgb_binary_pred", "XGBoost binary"),
            (DATA_DIR / "randomforest_binary_results.parquet", "rf_binary_pred", "Random Forest binary"),
            (
                DATA_DIR / "xgboost_per_attr_ensemble_results.parquet",
                "xgb_binary_from_attr",
                "XGBoost per-attr ensemble (binary)",
            ),
        ]:
            if not path.exists():
                continue
            try:
                df = read_parquet_safe(str(path))
                if pred_col not in df.columns:
                    continue
                m = merge_golden(df, pred_col=pred_col)
                if m.empty:
                    continue
                m["_pred"] = m[pred_col].astype(str).str.strip().str.lower()
                met = metrics_binary(m["_truth_2class"], m["_pred"])
                results_binary.append({"Method": name, "Variant": "raw", **met})
            except Exception as e:
                logger.warning("Binary %s: %s", name, e)

    # ----- Binary: SLM recalculated (one best row) and raw (3 models) -----
    if "_truth_2class" in golden.columns:
        def _three_to_binary(lbl) -> str:
            if pd.isna(lbl): return ""
            v = str(lbl).strip().lower()
            return "alt" if v in ("alt", "both") else "base"
        slm_binary_recalc_list: list[dict] = []
        # phase3_slm_labeled.parquet is labeled by Gemma3 4B (not GPT-4o mini)
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                df["id"] = df["id"].astype(str)
                attr_cols = [f"attr_{a}_winner" for a in ATTR_ATTRS]
                has_attr = all(c in df.columns for c in attr_cols)
                has_raw = "golden_label" in df.columns
                if has_attr:
                    df["_recalc_3class"] = df.apply(recalculate_3class_label, axis=1)
                    df["_recalc_binary"] = df["_recalc_3class"].apply(_three_to_binary)
                    m = golden.merge(df[["id", "_recalc_binary"]], on="id", how="inner")
                    if not m.empty:
                        met = metrics_binary(m["_truth_2class"], m["_recalc_binary"])
                        slm_binary_recalc_list.append(met)
                if has_raw:
                    df["_raw_binary"] = df["golden_label"].apply(
                        lambda x: "alt" if str(x).strip().lower() in ("alt", "both") else "base" if pd.notna(x) else ""
                    )
                    m = golden.merge(df[["id", "_raw_binary"]], on="id", how="inner")
                    if not m.empty:
                        met = metrics_binary(m["_truth_2class"], m["_raw_binary"])
                        results_binary.append({"Method": slm_name, "Variant": "raw", **met})
            except Exception as e:
                logger.warning("SLM binary %s: %s", slm_name, e)
        if slm_binary_recalc_list:
            best_recalc = max(slm_binary_recalc_list, key=lambda x: x.get("accuracy", 0))
            results_binary.append({"Method": "SLM Recalculated", "Variant": "recalculated", **best_recalc})

    # ----- 3-class raw: XGBoost 3-class, Phase5, SLM golden_label -----
    if "_truth_3class" in golden.columns:
        # XGBoost 3-class (match -> alt) and per-attribute ensemble 3-class
        for path, pred_col, name in [
            (DATA_DIR / "xgboost_results.parquet", "xgb_prediction", "XGBoost 3-class"),
            (
                DATA_DIR / "xgboost_per_attr_ensemble_results.parquet",
                "xgb_3class_from_attr",
                "XGBoost per-attr ensemble (3-class)",
            ),
        ]:
            if not path.exists():
                continue
            try:
                df = read_parquet_safe(str(path))
                if pred_col not in df.columns:
                    continue
                m = merge_golden(df, pred_col=pred_col)
                m = filter_3class_eval(m)
                if m.empty:
                    continue
                m["_pred"] = (
                    m[pred_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"match": "alt"})
                )
                met = metrics_3class(m["_truth_3class"], m["_pred"])
                results_3class.append({"Method": name, "Variant": "raw", **met})
            except Exception as e:
                logger.warning("%s: %s", name, e)

        # Phase5
        path = DATA_DIR / "phase5_full_results.parquet"
        if path.exists():
            for pred_col, name in [
                ("baseline_selection", "Phase5 confidence baseline"),
                ("model_3class_prediction", "Phase5 XGBoost 3-class"),
            ]:
                try:
                    df = read_parquet_safe(str(path))
                    if pred_col not in df.columns:
                        continue
                    m = merge_golden(df, pred_col=pred_col)
                    m = filter_3class_eval(m)
                    if m.empty:
                        continue
                    m["_pred"] = m[pred_col].astype(str).str.strip().str.lower()
                    met = metrics_3class(m["_truth_3class"], m["_pred"])
                    results_3class.append({"Method": name, "Variant": "raw", **met})
                except Exception as e:
                    logger.warning("Phase5 %s: %s", pred_col, e)

        # SLM: raw golden_label (map 4->3)
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                if "golden_label" not in df.columns:
                    continue
                m = merge_golden(df, pred_col="golden_label")
                m = filter_3class_eval(m)
                if not m.empty:
                    m["_pred"] = m["golden_label"].apply(
                        lambda x: fourclass_to_threeclass(str(x).strip().lower()) if pd.notna(x) else np.nan
                    )
                    met = metrics_3class(m["_truth_3class"], m["_pred"])
                    results_3class.append({"Method": slm_name, "Variant": "raw", **met})
            except Exception as e:
                logger.warning("SLM raw %s: %s", slm_name, e)

    # ----- 3-class recalculated: SLM attr_*_winner (one best "SLM Recalculated" row) -----
    if "_truth_3class" in golden.columns:
        slm_3class_recalc_list: list[dict] = []
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                attr_cols = [f"attr_{a}_winner" for a in ATTR_ATTRS]
                if not all(c in df.columns for c in attr_cols):
                    continue
                df = df.copy()
                df["_recalc_3class"] = df.apply(recalculate_3class_label, axis=1)
                df["_recalc_4class"] = df.apply(recalculate_4class_label, axis=1)
                m = merge_golden(df, pred_col="_recalc_3class", cols=["_recalc_3class", "_recalc_4class"])
                m = filter_3class_eval(m)
                if not m.empty:
                    met3 = metrics_3class(m["_truth_3class"], m["_recalc_3class"])
                    slm_3class_recalc_list.append(met3)
            except Exception as e:
                logger.warning("SLM recalc %s: %s", slm_name, e)
        if slm_3class_recalc_list:
            best_recalc = max(slm_3class_recalc_list, key=lambda x: x.get("accuracy", 0))
            results_3class.append({"Method": "SLM Recalculated", "Variant": "recalculated", **best_recalc})

    # SLM 4-class recalculated: merge synthetic with df on id
    if synthetic is not None and "_truth_4class" in synthetic.columns:
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                df["id"] = df["id"].astype(str)
                df["_recalc_4class"] = df.apply(recalculate_4class_label, axis=1)
                m = synthetic.merge(df[["id", "_recalc_4class"]], on="id", how="inner")
                if not m.empty:
                    met = metrics_4class(m["_truth_4class"], m["_recalc_4class"])
                    results_4class.append({"Method": slm_name, "Variant": "recalculated", **met})
            except Exception as e:
                logger.warning("SLM 4-class recalc %s: %s", slm_name, e)

    # ----- 4-class raw: XGBoost 4-class (model overall winner only) and per-attribute ensemble 4-class -----
    if synthetic is not None and "_truth_4class" in synthetic.columns:
        for path, pred_col, name in [
            (DATA_DIR / "xgboost_multiclass_results.parquet", "xgb_4class_pred", "XGBoost 4-class"),
            (
                DATA_DIR / "xgboost_per_attr_ensemble_results.parquet",
                "xgb_4class_from_attr",
                "XGBoost per-attr ensemble (4-class)",
            ),
        ]:
            if not path.exists():
                continue
            try:
                df = read_parquet_safe(str(path))
                if pred_col not in df.columns:
                    continue
                df["id"] = df["id"].astype(str)
                m = synthetic.merge(df[["id", pred_col]], on="id", how="inner")
                m["_pred"] = m[pred_col].astype(str).str.strip().str.lower()
                met = metrics_4class(m["_truth_4class"], m["_pred"])
                results_4class.append({"Method": name, "Variant": "raw", **met})
            except Exception as e:
                logger.warning("%s: %s", name, e)

        # SLM raw 4-class (golden_label)
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                if "golden_label" not in df.columns:
                    continue
                df["id"] = df["id"].astype(str)
                df["_raw_4"] = df["golden_label"].astype(str).str.strip().str.lower()
                m = synthetic.merge(df[["id", "_raw_4"]], on="id", how="inner")
                met = metrics_4class(m["_truth_4class"], m["_raw_4"])
                results_4class.append({"Method": slm_name, "Variant": "raw", **met})
            except Exception as e:
                logger.warning("SLM 4-class raw %s: %s", slm_name, e)

    # ----- Rule-based, Google, Scrape: truth_*_winner -> recalculated 3 and 4 -----
    for path, name in [
        (DATA_DIR / "rule_based.parquet", "Rule-based"),
        (DATA_DIR / "ground_truth_google_golden.parquet", "Google API"),
        (DATA_DIR / "ground_truth_scrape_golden.parquet", "Scrape + search"),
    ]:
        if not path.exists():
            continue
        try:
            df = read_parquet_safe(str(path))
            if not all(c in df.columns for c in TRUTH_WINNER_COLS):
                continue
            df = df.copy()
            df["id"] = df["id"].astype(str)
            df["_recalc_3class"] = df.apply(truth_to_3class, axis=1)
            df["_recalc_4class"] = df.apply(truth_to_4class, axis=1)
            if "_truth_3class" in golden.columns:
                m = golden.merge(df[["id", "_recalc_3class"]], on="id", how="inner")
                m = filter_3class_eval(m)
                if not m.empty:
                    met3 = metrics_3class(m["_truth_3class"], m["_recalc_3class"])
                    results_3class.append({"Method": name, "Variant": "recalculated", **met3})
            if synthetic is not None and "_truth_4class" in synthetic.columns:
                m = synthetic.merge(df[["id", "_recalc_4class"]], on="id", how="inner")
                if not m.empty:
                    met4 = metrics_4class(m["_truth_4class"], m["_recalc_4class"])
                    results_4class.append({"Method": name, "Variant": "recalculated", **met4})
        except Exception as e:
            logger.warning("%s: %s", name, e)

    # ----- Similarity (Google, Scrape): value similarity vs golden -----
    try:
        from external_validation.verify_truth import ATTR_MAP, _golden_value, _similarity
        from external_validation.verify_truth import _norm_label as _ev_norm_label
        from scripts.normalization import (
            standardize_phone,
            normalize_website,
            normalize_address_json,
            normalize_address_json_full,
        )

        def _norm_val(attr: str, val) -> str:
            """Normalize value for comparison: phone standardized, web normalized, address full form, category lower."""
            if attr == "phone":
                return standardize_phone(val) or ""
            if attr == "web":
                return normalize_website(val) or ""
            if attr == "address":
                return normalize_address_json_full(val) or ""
            return (str(val) or "").strip().lower() if pd.notna(val) else ""

        def _norm_val_scrape_address(val) -> str:
            """Scrape-only address normalization (legacy behavior): use normalize_address_json, not full address."""
            return normalize_address_json(val) or ""

        def _golden_value_norm(attr_key: str, row: pd.Series, golden_winner_col: str, base_col: str, alt_col: str) -> str:
            """Golden value normalized for comparison. For address, use full address from golden record."""
            if attr_key == "address":
                # Prefer raw address columns so we build full address (freeform + locality + region + postcode + country)
                if "base_addresses" in row.index and "addresses" in row.index:
                    w = _ev_norm_label(row.get(golden_winner_col))
                    if w in ("base", "none"):
                        raw = row.get("base_addresses")
                    elif w == "alt":
                        raw = row.get("addresses")
                    else:
                        raw = row.get("base_addresses")  # both: prefer base
                    return normalize_address_json_full(raw) or ""
                # Fallback: norm_base_addr / norm_conflated_addr (may already be full if from process_addresses)
                raw = _golden_value(row, golden_winner_col, base_col, alt_col)
                return normalize_address_json_full(raw) if raw else ""
            raw = _golden_value(row, golden_winner_col, base_col, alt_col)
            return _norm_val(attr_key, raw)

        def _golden_value_norm_scrape_address(row: pd.Series, golden_winner_col: str, base_col: str, alt_col: str) -> str:
            """Scrape-only golden address normalization: legacy behavior using normalize_address_json."""
            raw = _golden_value(row, golden_winner_col, base_col, alt_col)
            return normalize_address_json(raw) if raw else ""

        for path, name in [
            (DATA_DIR / "ground_truth_google_golden.parquet", "Google API"),
            (DATA_DIR / "ground_truth_scrape_golden.parquet", "Scrape + search"),
        ]:
            if not path.exists():
                continue
            try:
                df = read_parquet_safe(str(path))
                needed = [t[0] for t in ATTR_MAP] + [t[1] for t in ATTR_MAP] + [t[2] for t in ATTR_MAP]
                if not all(c in df.columns for c in needed) or "attr_phone_winner" not in golden.columns:
                    continue
                m = golden.merge(df[["id"] + [t[0] for t in ATTR_MAP] + [t[1] for t in ATTR_MAP]], on="id", how="inner")
                if m.empty:
                    continue
                sims = []
                attr_keys = ["phone", "web", "address", "category"]
                for attr_key, (truth_winner_col, truth_value_col, golden_winner_col, base_col, alt_col) in zip(
                    attr_keys, ATTR_MAP
                ):
                    if name == "Scrape + search" and attr_key == "address":
                        # Scrape-only: legacy address similarity using normalize_address_json for both sides
                        s = m.apply(
                            lambda r: _similarity(
                                attr_key,
                                _norm_val_scrape_address(r.get(truth_value_col)),
                                _golden_value_norm_scrape_address(r, golden_winner_col, base_col, alt_col),
                            ),
                            axis=1,
                        )
                    else:
                        s = m.apply(
                            lambda r: _similarity(
                                attr_key,
                                _norm_val(attr_key, r.get(truth_value_col)),
                                _golden_value_norm(attr_key, r, golden_winner_col, base_col, alt_col),
                            ),
                            axis=1,
                        )
                    sims.append(s.mean())
                # Overall similarity: phone, web, address only (exclude category)
                mean_overall = float(np.mean(sims[:3])) if len(sims) >= 3 else (float(np.mean(sims)) if sims else 0.0)
                row = {"Method": name, "mean_overall_similarity": mean_overall, "n": len(m)}
                for k, mean_sim in zip(attr_keys, sims):
                    row[f"{k}_sim"] = float(mean_sim)
                similarity_rows.append(row)
            except Exception as e:
                logger.warning("Similarity %s: %s", name, e)
    except ImportError:
        pass

    # ----- Per-attribute agreement (golden attr_*_winner vs method truth_*_winner) -----
    # External validation has truth_phone_winner, truth_web_winner, truth_address_winner, truth_category_winner (no name)
    for path, name in [
        (DATA_DIR / "rule_based.parquet", "Rule-based"),
        (DATA_DIR / "ground_truth_google_golden.parquet", "Google API"),
        (DATA_DIR / "ground_truth_scrape_golden.parquet", "Scrape + search"),
    ]:
        if not path.exists():
            continue
        try:
            df = read_parquet_safe(str(path))
            df["id"] = df["id"].astype(str)
            truth_cols = [c for c in TRUTH_WINNER_COLS if c in df.columns]
            if not truth_cols:
                continue
            m = golden.merge(df[["id"] + truth_cols], on="id", how="inner")
            if m.empty:
                continue
            agree_total = 0
            total_count = 0
            acc_by_attr: dict[str, float] = {}
            for tc in truth_cols:
                attr = tc.replace("truth_", "").replace("_winner", "")
                gc = f"attr_{attr}_winner"
                if gc not in m.columns:
                    continue
                # Normalize: method may use "real" for alt
                golden_norm = m[gc].astype(str).apply(lambda x: _norm_attr_winner_for_compare(x))
                method_norm = m[tc].astype(str).apply(lambda x: _norm_attr_winner_for_compare(x))
                agree = (golden_norm == method_norm).sum()
                agree_total += agree
                total_count += len(m)
                acc_by_attr[f"{attr}_acc"] = 100.0 * agree / len(m) if len(m) else 0.0
            if total_count:
                per_attr_rows.append({"Method": name, "agreement_pct": 100.0 * agree_total / total_count, "n": len(m)})
            if acc_by_attr:
                per_attr_accuracy_rows.append({"Method": name, "n": len(m), **acc_by_attr})
        except Exception as e:
            logger.warning("Per-attr %s: %s", name, e)

    # ----- Per-attribute accuracy: SLM and XGBoost per-attr ensemble (attr_*_winner vs golden attr_*_winner) -----
    if all(f"attr_{a}_winner" in golden.columns for a in PER_ATTR_ACCURACY_ATTRS):
        # SLM models
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                df["id"] = df["id"].astype(str)
                attr_cols = [f"attr_{a}_winner" for a in PER_ATTR_ACCURACY_ATTRS]
                if not all(c in df.columns for c in attr_cols):
                    continue
                m = golden.merge(df[["id"] + attr_cols], on="id", how="inner")
                if m.empty:
                    continue
                acc_by_attr = {}
                for attr in PER_ATTR_ACCURACY_ATTRS:
                    gc = f"attr_{attr}_winner"
                    col_golden = gc + "_x" if gc + "_x" in m.columns else gc
                    col_method = gc + "_y" if gc + "_y" in m.columns else gc
                    golden_norm = m[col_golden].astype(str).apply(_norm_attr_winner_for_compare)
                    method_norm = m[col_method].astype(str).apply(_norm_attr_winner_for_compare)
                    agree = (golden_norm == method_norm).sum()
                    acc_by_attr[f"{attr}_acc"] = 100.0 * agree / len(m)
                per_attr_accuracy_rows.append({"Method": slm_name, "n": len(m), **acc_by_attr})
            except Exception as e:
                logger.warning("Per-attr accuracy SLM %s: %s", slm_name, e)

        # XGBoost per-attribute ensemble (uses attr_*_winner_xgb vs golden attr_*_winner)
        xgb_attr_path = DATA_DIR / "xgboost_per_attr_ensemble_results.parquet"
        if xgb_attr_path.exists():
            try:
                df = read_parquet_safe(str(xgb_attr_path))
                df["id"] = df["id"].astype(str)
                attr_cols = [f"attr_{a}_winner_xgb" for a in PER_ATTR_ACCURACY_ATTRS]
                if all(c in df.columns for c in attr_cols):
                    m = golden.merge(df[["id"] + attr_cols], on="id", how="inner")
                    if not m.empty:
                        acc_by_attr = {}
                        for attr in PER_ATTR_ACCURACY_ATTRS:
                            gc = f"attr_{attr}_winner"
                            col_golden = gc
                            col_method = f"attr_{attr}_winner_xgb"
                            if col_golden not in m.columns or col_method not in m.columns:
                                continue
                            golden_norm = m[col_golden].astype(str).apply(_norm_attr_winner_for_compare)
                            method_norm = m[col_method].astype(str).apply(_norm_attr_winner_for_compare)
                            agree = (golden_norm == method_norm).sum()
                            acc_by_attr[f"{attr}_acc"] = 100.0 * agree / len(m)
                        if acc_by_attr:
                            per_attr_accuracy_rows.append(
                                {"Method": "XGBoost per-attr ensemble", "n": len(m), **acc_by_attr}
                            )
            except Exception as e:
                logger.warning("Per-attr accuracy XGBoost per-attr ensemble: %s", e)

    # SLM binary with truth = 3-class → binary (for reference-style table)
    if "_truth_2class_from_3class" in golden.columns:
        def _three_to_binary(lbl) -> str:
            if pd.isna(lbl): return ""
            v = str(lbl).strip().lower()
            return "alt" if v in ("alt", "both") else "base"
        for slm_path, slm_name in [
            (DATA_DIR / "phase3_slm_labeled.parquet", "SLM (Gemma3 4B)"),
            (DATA_DIR / "phase3_slm_labeledkimi.parquet", "SLM (Kimi)"),
            (DATA_DIR / "phase3_slm_labeledgpt4omini.parquet", "SLM (GPT-4o mini)"),
        ]:
            if not slm_path.exists():
                continue
            try:
                df = read_parquet_safe(str(slm_path))
                df["id"] = df["id"].astype(str)
                attr_cols = [f"attr_{a}_winner" for a in ATTR_ATTRS]
                has_attr = all(c in df.columns for c in attr_cols)
                has_raw = "golden_label" in df.columns
                m = golden.merge(df[["id"]], on="id", how="inner")
                if m.empty:
                    continue
                if has_attr:
                    df["_recalc_3"] = df.apply(recalculate_3class_label, axis=1)
                    df["_recalc_b"] = df["_recalc_3"].apply(_three_to_binary)
                    m = golden.merge(df[["id", "_recalc_b"]], on="id", how="inner")
                    m = filter_3class_eval(m)
                    if not m.empty:
                        acc = (m["_truth_2class_from_3class"] == m["_recalc_b"]).mean()
                        key = "recalc_b_" + slm_name.replace(" ", "_").replace("(", "").replace(")", "")
                        slm_binary_3class_truth[key] = float(acc) * 100
                if has_raw:
                    df["_raw_b"] = df["golden_label"].apply(
                        lambda x: "alt" if str(x).strip().lower() in ("alt", "both") else "base" if pd.notna(x) else ""
                    )
                    m = golden.merge(df[["id", "_raw_b"]], on="id", how="inner")
                    m = filter_3class_eval(m)
                    if not m.empty:
                        acc = (m["_truth_2class_from_3class"] == m["_raw_b"]).mean()
                        key = "raw_b_" + slm_name.replace(" ", "_").replace("(", "").replace(")", "")
                        slm_binary_3class_truth[key] = float(acc) * 100
            except Exception as e:
                logger.warning("SLM binary (3class truth) %s: %s", slm_name, e)
        if slm_binary_3class_truth:
            recalc_b_list = [v for k, v in slm_binary_3class_truth.items() if k.startswith("recalc_b_") and k != "recalc_b_max"]
            if recalc_b_list:
                slm_binary_3class_truth["recalc_b_max"] = max(recalc_b_list)

    return results_3class, results_4class, results_binary, similarity_rows, per_attr_rows, per_attr_accuracy_rows, slm_binary_3class_truth


def compute_readability(golden_df: pd.DataFrame) -> dict:
    """Base vs alt completeness on golden: mean non-empty count for phone, web, address, category."""
    out = {"base_mean_filled": np.nan, "alt_mean_filled": np.nan, "n": 0}
    base_names = ["norm_base_phone", "norm_base_website", "norm_base_addr", "_base_category"]
    alt_names = ["norm_conflated_phone", "norm_conflated_website", "norm_conflated_addr", "_category"]
    # Fallbacks
    if "norm_base_website" not in golden_df.columns and "base_websites" in golden_df.columns:
        base_names = ["norm_base_phone", "base_websites", "norm_base_addr", "_base_category"]
    if "norm_conflated_website" not in golden_df.columns and "websites" in golden_df.columns:
        alt_names = ["norm_conflated_phone", "websites", "norm_conflated_addr", "_category"]
    base_names = [c for c in base_names if c in golden_df.columns]
    alt_names = [c for c in alt_names if c in golden_df.columns]
    if not base_names or not alt_names or len(golden_df) == 0:
        out["n"] = len(golden_df)
        return out

    def _filled(row, names):
        return sum(1 for c in names if pd.notna(row.get(c)) and str(row.get(c)).strip())

    out["base_mean_filled"] = float(golden_df.apply(lambda r: _filled(r, base_names), axis=1).mean())
    out["alt_mean_filled"] = float(golden_df.apply(lambda r: _filled(r, alt_names), axis=1).mean())
    out["n"] = len(golden_df)
    return out


def write_report(
    results_3class: list,
    results_4class: list,
    results_binary: list,
    similarity_rows: list,
    per_attr_rows: list,
    per_attr_accuracy_rows: list,
    readability: dict | None = None,
    slm_binary_3class_truth: dict | None = None,
    report_suffix: str = "",
) -> None:
    """Write metrics summary markdown report. report_suffix e.g. '_golden100' → metrics_summary_golden100.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if report_suffix:
        path = REPORTS_DIR / f"metrics_summary{report_suffix}.md"
    else:
        path = REPORTS_DIR / "metrics_summary_golden200.md"
    title = "Golden 100" if report_suffix == "_golden100" else "Golden 200 / Synthetic 4-Class"
    lines = [
        f"# Unified Metrics Summary ({title})",
        "",
        "## 3-Class Metrics (vs golden_dataset_200)",
        "",
        "*(Excluding rows where golden 3-class label is \"none\"; only alt / both / base.)*",
        "",
    ]
    if results_3class:
        df3 = pd.DataFrame(results_3class)
        lines.append("```")
        lines.append(df3.to_string(index=False))
        lines.append("```")
        lines.append("")
        if "f1_macro" in df3.columns and df3["f1_macro"].notna().any():
            best_f1_3 = df3.loc[df3["f1_macro"].idxmax()]
            lines.append(f"- **Best F1 (macro) 3-class:** {best_f1_3.get('Method', '')} ({best_f1_3.get('Variant', '')}) = {best_f1_3.get('f1_macro', 0):.4f}")
        if "accuracy" in df3.columns and df3["accuracy"].notna().any():
            best_acc_3 = df3.loc[df3["accuracy"].idxmax()]
            lines.append(f"- **Best Accuracy 3-class:** {best_acc_3.get('Method', '')} ({best_acc_3.get('Variant', '')}) = {best_acc_3.get('accuracy', 0):.4f}")
    else:
        lines.append("No 3-class results.")
    lines.append("")
    lines.append("## 4-Class Metrics (vs synthetic_4class_golden)")
    lines.append("")
    if results_4class:
        df4 = pd.DataFrame(results_4class)
        lines.append("```")
        lines.append(df4.to_string(index=False))
        lines.append("```")
        lines.append("")
        if "f1_macro" in df4.columns and df4["f1_macro"].notna().any():
            best_f1_4 = df4.loc[df4["f1_macro"].idxmax()]
            lines.append(f"- **Best F1 (macro) 4-class:** {best_f1_4.get('Method', '')} ({best_f1_4.get('Variant', '')}) = {best_f1_4.get('f1_macro', 0):.4f}")
        if "accuracy" in df4.columns and df4["accuracy"].notna().any():
            best_acc_4 = df4.loc[df4["accuracy"].idxmax()]
            lines.append(f"- **Best Accuracy 4-class:** {best_acc_4.get('Method', '')} ({best_acc_4.get('Variant', '')}) = {best_acc_4.get('accuracy', 0):.4f}")
    else:
        lines.append("No 4-class results.")
    lines.append("")
    lines.append("## Binary Metrics (vs golden_dataset_200 2class_testlabels)")
    lines.append("")
    if results_binary:
        dfb = pd.DataFrame(results_binary)
        lines.append("```")
        lines.append(dfb.to_string(index=False))
        lines.append("```")
        lines.append("")
        if "f1" in dfb.columns and dfb["f1"].notna().any():
            best_f1_b = dfb.loc[dfb["f1"].idxmax()]
            lines.append(f"- **Best F1 (binary):** {best_f1_b.get('Method', '')} = {best_f1_b.get('f1', 0):.4f}")
    else:
        lines.append("No binary results.")
    lines.append("")

    lines.append("## Similarity (value similarity vs golden)")
    lines.append("")
    if similarity_rows:
        lines.append("```")
        sim_df = pd.DataFrame(similarity_rows)
        # Overall only for this section
        overall_cols = [c for c in ["Method", "mean_overall_similarity", "n"] if c in sim_df.columns]
        lines.append(sim_df[overall_cols].to_string(index=False))
        lines.append("```")
        # Per-attribute similarity (0–100 mean per attr)
        sim_attr_cols = [c for c in ["phone_sim", "web_sim", "address_sim", "category_sim"] if c in sim_df.columns]
        if sim_attr_cols:
            lines.append("")
            lines.append("### Per-Attribute Similarity")
            lines.append("")
            lines.append("Mean value similarity (0–100) vs golden per attribute.")
            lines.append("")
            lines.append("```")
            per_attr_sim = sim_df[["Method"] + sim_attr_cols + (["n"] if "n" in sim_df.columns else [])].copy()
            for c in sim_attr_cols:
                per_attr_sim[c] = per_attr_sim[c].round(2)
            per_attr_sim = per_attr_sim.rename(columns={c: c.replace("_sim", "") for c in sim_attr_cols})
            lines.append(per_attr_sim.to_string(index=False))
            lines.append("```")
    else:
        lines.append("No similarity metrics.")
    lines.append("")
    lines.append("## Per-Attribute Winner Agreement")
    lines.append("")
    if per_attr_rows:
        lines.append("```")
        lines.append(pd.DataFrame(per_attr_rows).to_string(index=False))
        lines.append("```")
    else:
        lines.append("No per-attribute agreement.")
    lines.append("")
    lines.append("## Per-Attribute Accuracy")
    lines.append("")
    lines.append("Accuracy per attribute (method winner vs golden `attr_*_winner`) for methods with per-attribute outputs.")
    lines.append("")
    if per_attr_accuracy_rows:
        acc_df = pd.DataFrame(per_attr_accuracy_rows)
        # Order columns: Method, phone_acc, web_acc, address_acc, category_acc, n
        acc_cols = [c for c in ["phone_acc", "web_acc", "address_acc", "category_acc"] if c in acc_df.columns]
        if acc_cols:
            disp = acc_df[["Method"] + acc_cols + (["n"] if "n" in acc_df.columns else [])].copy()
            for c in acc_cols:
                if disp[c].dtype in (np.floating, float):
                    disp[c] = disp[c].round(1).astype(str) + "%"
            lines.append("```")
            lines.append(disp.to_string(index=False))
            lines.append("```")
        else:
            lines.append(pd.DataFrame(per_attr_accuracy_rows).to_string(index=False))
    else:
        lines.append("No per-attribute accuracy (no methods with per-attribute outputs).")
    lines.append("")
    lines.append("## Readability / Informativeness")
    lines.append("")
    if readability and not np.isnan(readability.get("base_mean_filled", np.nan)):
        lines.append("Mean number of filled attributes (phone, web, address, category) per row on golden:")
        lines.append(f"- **Base (source):** {readability.get('base_mean_filled', 0):.2f}")
        lines.append(f"- **Alt (conflated):** {readability.get('alt_mean_filled', 0):.2f}")
        b = readability.get("base_mean_filled") or 0
        a = readability.get("alt_mean_filled") or 0
        lines.append(f"- **Which has more information:** {'Alt' if a >= b else 'Base'}")
        lines.append("")
    else:
        lines.append("Base vs alt completeness on golden rows: run with golden_dataset_200 to compute mean filled-attribute count for base vs conflated/alt.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified metrics vs golden 200 and synthetic 4-class")
    parser.add_argument("--report", action="store_true", help="Write metrics summary markdown report")
    parser.add_argument("--golden", type=str, default=None, help="Path to golden_dataset_200.parquet")
    parser.add_argument("--synthetic", type=str, default=None, help="Path to synthetic_4class_golden.parquet")
    parser.add_argument("--golden-limit", type=int, default=None, help="Use only first N rows of golden (e.g. 100); report/CSVs use _goldenN suffix")
    args = parser.parse_args()
    global GOLDEN_200_PATH, SYNTHETIC_4CLASS_PATH
    if args.golden:
        GOLDEN_200_PATH = Path(args.golden)
    if args.synthetic:
        SYNTHETIC_4CLASS_PATH = Path(args.synthetic)

    golden = load_golden_200()
    if args.golden_limit is not None:
        golden = golden.head(args.golden_limit).copy()
        logger.info("Using first %d rows of golden for measurements.", len(golden))
    report_suffix = f"_golden{args.golden_limit}" if args.golden_limit is not None else ""
    synthetic = load_synthetic_4class()
    results_3class, results_4class, results_binary, similarity_rows, per_attr_rows, per_attr_accuracy_rows, slm_binary_3class_truth = run_evaluations(golden, synthetic)
    readability = compute_readability(golden)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suf = report_suffix  # e.g. _golden100 for separate output files
    if results_3class:
        df3 = pd.DataFrame(results_3class)
        out = REPORTS_DIR / f"unified_metrics_3class{suf}.csv"
        df3.to_csv(out, index=False)
        logger.info("Wrote %s", out)
    if results_4class:
        df4 = pd.DataFrame(results_4class)
        out = REPORTS_DIR / f"unified_metrics_4class{suf}.csv"
        df4.to_csv(out, index=False)
        logger.info("Wrote %s", out)
    if results_binary:
        dfb = pd.DataFrame(results_binary)
        out = REPORTS_DIR / f"unified_metrics_binary{suf}.csv"
        dfb.to_csv(out, index=False)
        logger.info("Wrote %s", out)
    if similarity_rows:
        pd.DataFrame(similarity_rows).to_csv(REPORTS_DIR / f"unified_metrics_similarity{suf}.csv", index=False)
    if per_attr_rows:
        pd.DataFrame(per_attr_rows).to_csv(REPORTS_DIR / f"unified_metrics_per_attr_agreement{suf}.csv", index=False)
    if per_attr_accuracy_rows:
        pd.DataFrame(per_attr_accuracy_rows).to_csv(REPORTS_DIR / f"unified_metrics_per_attr_accuracy{suf}.csv", index=False)

    if args.report:
        write_report(results_3class, results_4class, results_binary, similarity_rows, per_attr_rows, per_attr_accuracy_rows, readability, slm_binary_3class_truth, report_suffix)
    else:
        logger.info("Run with --report to generate reports/metrics_summary_golden200.md")


if __name__ == "__main__":
    main()
