from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.labels import ATTR_ATTRS, fourclass_to_threeclass, recalculate_3class_label, recalculate_4class_label
from scripts.parquet_io import read_parquet_safe
from scripts.unified_metrics_golden200 import (
    load_golden_200,
    load_synthetic_4class,
    metrics_3class,
    metrics_4class,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def metrics_binary_safe(truth: pd.Series, pred: pd.Series) -> dict:
    """Binary metrics where labels are coerced to {'alt','base'} and others dropped."""
    t = truth.astype(str).str.strip().str.lower()
    p = pred.astype(str).str.strip().str.lower()

    def to_bin(x: str) -> str:
        if x == "alt":
            return "alt"
        if x == "base":
            return "base"
        return ""

    t_b = t.map(to_bin)
    p_b = p.map(to_bin)
    mask = (t_b.isin({"alt", "base"})) & (p_b.isin({"alt", "base"}))
    t_f = t_b[mask]
    p_f = p_b[mask]
    if len(t_f) == 0:
        return {"accuracy": np.nan, "f1": np.nan, "n": 0}
    acc = (t_f == p_f).mean()
    tp = ((t_f == "alt") & (p_f == "alt")).sum()
    fp = ((t_f == "base") & (p_f == "alt")).sum()
    fn = ((t_f == "alt") & (p_f == "base")).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": float(acc), "f1": float(f1), "n": int(len(t_f))}


def main() -> None:
    slm_name = "SLM (Gemma3 4B, 2-prompt)"
    slm_path = DATA_DIR / "phase3_slm_labeledgemma3_4B.parquet"

    if not slm_path.exists():
        raise SystemExit(f"Missing parquet: {slm_path}")

    print(f"Using SLM file: {slm_path}")

    golden = load_golden_200()
    synthetic = load_synthetic_4class()

    valid_3 = {"alt", "both", "base"}
    if "_truth_3class" in golden.columns:
        mask_3 = (
            golden["_truth_3class"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(valid_3)
        )
    else:
        mask_3 = pd.Series(False, index=golden.index)

    slm = read_parquet_safe(str(slm_path))
    slm["id"] = slm["id"].astype(str)

    results_3: list[tuple[str, dict]] = []
    results_4: list[tuple[str, dict]] = []
    results_bin: list[tuple[str, dict]] = []

    # --- Raw 3-class from golden_label (4->3) ---
    if "golden_label" in slm.columns and "_truth_3class" in golden.columns:
        m = golden[mask_3].merge(slm[["id", "golden_label"]], on="id", how="inner")
        if not m.empty:
            m["_pred3"] = m["golden_label"].apply(
                lambda x: fourclass_to_threeclass(str(x).strip().lower())
                if pd.notna(x)
                else np.nan
            )
            met3 = metrics_3class(m["_truth_3class"], m["_pred3"])
            results_3.append(("raw_from_golden_label", met3))

    # --- Recalculated 3-class & 4-class from attr_*_winner ---
    attr_cols = [f"attr_{a}_winner" for a in ATTR_ATTRS]
    if all(c in slm.columns for c in attr_cols) and "_truth_3class" in golden.columns:
        slm_rc = slm.copy()
        slm_rc["_recalc_3"] = slm_rc.apply(recalculate_3class_label, axis=1)
        slm_rc["_recalc_4"] = slm_rc.apply(recalculate_4class_label, axis=1)

        m3 = golden[mask_3].merge(
            slm_rc[["id", "_recalc_3"]], on="id", how="inner"
        )
        if not m3.empty:
            met3_rc = metrics_3class(m3["_truth_3class"], m3["_recalc_3"])
            results_3.append(("recalculated_from_attrs", met3_rc))

        if synthetic is not None and "_truth_4class" in synthetic.columns:
            m4 = synthetic.merge(
                slm_rc[["id", "_recalc_4"]], on="id", how="inner"
            )
            if not m4.empty:
                met4_rc = metrics_4class(m4["_truth_4class"], m4["_recalc_4"])
                results_4.append(("recalculated_from_attrs", met4_rc))

    # --- Raw binary from golden_label (alt+both vs base/none) ---
    if "golden_label" in slm.columns and "_truth_2class" in golden.columns:

        def four_to_bin(v: str) -> str:
            v = (v or "").strip().lower()
            if v in ("alt", "both"):
                return "alt"
            if v in ("base", "none"):
                return "base"
            return ""

        slm_bin = slm[["id", "golden_label"]].copy()
        slm_bin["_raw_b"] = slm_bin["golden_label"].astype(str).map(four_to_bin)
        m = golden.merge(slm_bin[["id", "_raw_b"]], on="id", how="inner")
        if not m.empty:
            metb = metrics_binary_safe(m["_truth_2class"], m["_raw_b"])
            results_bin.append(("raw_from_golden_label", metb))

    # --- Recalculated binary from attr_*_winner via recalculated 3-class ---
    if all(c in slm.columns for c in attr_cols) and "_truth_2class" in golden.columns:

        def three_to_bin(v: str) -> str:
            v = (v or "").strip().lower()
            if v in ("alt", "both"):
                return "alt"
            if v == "base":
                return "base"
            return ""

        slm_rc2 = slm.copy()
        slm_rc2["_recalc_3"] = slm_rc2.apply(recalculate_3class_label, axis=1)
        slm_rc2["_recalc_b"] = slm_rc2["_recalc_3"].astype(str).map(three_to_bin)
        m = golden.merge(slm_rc2[["id", "_recalc_b"]], on="id", how="inner")
        if not m.empty:
            metb_rc = metrics_binary_safe(m["_truth_2class"], m["_recalc_b"])
            results_bin.append(("recalculated_from_attrs", metb_rc))

    print(f"\n=== Metrics for {slm_name} ===")

    if results_3:
        print("\n3-class metrics (vs golden_dataset_200):")
        for name, met in results_3:
            print(
                f"- {name}: accuracy={met['accuracy']:.4f}, "
                f"f1_macro={met['f1_macro']:.4f}, n={met['n']}"
            )
    else:
        print("No 3-class metrics computed.")

    if results_4:
        print("\n4-class metrics (vs synthetic_4class_golden):")
        for name, met in results_4:
            print(
                f"- {name}: accuracy={met['accuracy']:.4f}, "
                f"f1_macro={met['f1_macro']:.4f}, n={met['n']}"
            )
    else:
        print("No 4-class metrics computed.")

    if results_bin:
        print("\nBinary metrics (vs golden_dataset_200 2class_testlabels):")
        for name, met in results_bin:
            print(
                f"- {name}: accuracy={met['accuracy']:.4f}, "
                f"f1={met['f1']:.4f}, n={met['n']}"
            )
    else:
        print("No binary metrics computed.")


if __name__ == "__main__":
    main()

