"""
Re-apply the phone number logic (with country-code fallback and E164 comparison)
to every row in data/golden_dataset_100.parquet.
If the new phone winner differs from the stored one, update attr_phone_winner and
recompute golden_label from the five attribute scores; update golden_label if it changes.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from parquet_io import read_parquet_safe
from phonenumber_validator import validate_phone_number, try_with_region

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_dataset_100.parquet"
ATTR_ATTRS = ("name", "phone", "web", "address", "category")


def _region_from_address(addr):
    if addr is None or (isinstance(addr, float) and pd.isna(addr)):
        return None
    text = str(addr).upper()
    if " UNITED STATES" in text or " USA" in text or " U.S." in text or " U.S.A" in text:
        return "US"
    if " CANADA" in text or " CAN" in text or " ON " in text or " BC " in text or " AB " in text or " QB " in text:
        return "CA"
    if " UNITED KINGDOM" in text or " UK" in text or " GB " in text or " ENGLAND" in text or " SCOTLAND" in text:
        return "GB"
    if re.search(r"\b(AK|AL|AR|AZ|CA|CO|CT|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)\b", text):
        return "US"
    return None


def _first_str(val):
    if isinstance(val, list) and len(val) > 0:
        return str(val[0]).strip() or None
    return val


def _phone_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none") else None


def compute_phone_winner_and_corrected(row):
    """
    Phone logic: 1) validate with phonenumber_validator; 2) if invalid, try country code;
    3) both valid → both, none valid → none, only one valid → that one.
    Returns (phone_winner_str, corrected_phones_dict).
    """
    base_phone = _phone_str(_first_str(row.get("base_phone") or row.get("norm_base_phone") or row.get("base_phones")))
    alt_phone = _phone_str(_first_str(row.get("alt_phone") or row.get("norm_conflated_phone") or row.get("phones")))
    base_addr = row.get("base_address") or row.get("norm_base_addr") or row.get("base_addresses")
    alt_addr = row.get("alt_address") or row.get("norm_conflated_addr") or row.get("addresses")
    base_region = _region_from_address(base_addr) or "US"
    alt_region = _region_from_address(alt_addr) or "US"

    base_valid = validate_phone_number(base_phone)[0] if base_phone else False
    alt_valid = validate_phone_number(alt_phone)[0] if alt_phone else False
    corrected_phones = {}
    if base_phone and not base_valid:
        ok, e164 = try_with_region(base_phone, base_region)
        if ok:
            corrected_phones["base_phone_with_country"] = e164
            base_valid = True
    if alt_phone and not alt_valid:
        ok, e164 = try_with_region(alt_phone, alt_region)
        if ok:
            corrected_phones["alt_phone_with_country"] = e164
            alt_valid = True

    if base_valid and alt_valid:
        phone_winner = "both"
    elif not base_valid and not alt_valid:
        phone_winner = "none"
    elif base_valid:
        phone_winner = "base"
    else:
        phone_winner = "alt"
    return phone_winner, corrected_phones


def scores_to_label(base_score, alt_score):
    if base_score == 0 and alt_score >= 2:
        return "alt"
    if alt_score == 0 and base_score >= 2:
        return "base"
    if base_score >= alt_score + 2:
        return "base"
    if alt_score >= base_score + 2:
        return "alt"
    return "abstain"


def main():
    if not GOLDEN_PATH.exists():
        print(f"Error: {GOLDEN_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    df = read_parquet_safe(str(GOLDEN_PATH))
    if "attr_phone_winner" not in df.columns or "golden_label" not in df.columns:
        print("Error: golden dataset must have attr_phone_winner and golden_label columns.", file=sys.stderr)
        sys.exit(1)

    n_phone_changes = 0
    n_label_changes = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        new_phone_winner, corrected_phones = compute_phone_winner_and_corrected(row)
        old_phone_winner = row.get("attr_phone_winner")
        if pd.isna(old_phone_winner):
            old_phone_winner = "none"
        else:
            old_phone_winner = str(old_phone_winner).strip().lower()

        # Always apply new phone logic: update attr_phone_winner and corrected columns
        df.at[idx, "attr_phone_winner"] = new_phone_winner
        if "base_phone_with_country" in corrected_phones:
            df.at[idx, "base_phone_with_country"] = corrected_phones["base_phone_with_country"]
        if "alt_phone_with_country" in corrected_phones:
            df.at[idx, "alt_phone_with_country"] = corrected_phones["alt_phone_with_country"]

        if new_phone_winner != old_phone_winner:
            n_phone_changes += 1
            # Recompute scores from the five attributes and update golden_label if it changes
            winners = {
                "name": str(row.get("attr_name_winner", "none")).strip().lower() or "none",
                "phone": new_phone_winner,
                "web": str(row.get("attr_web_winner", "none")).strip().lower() or "none",
                "address": str(row.get("attr_address_winner", "none")).strip().lower() or "none",
                "category": str(row.get("attr_category_winner", "none")).strip().lower() or "none",
            }
            base_score = sum(1 for a in ATTR_ATTRS if winners[a] == "base")
            alt_score = sum(1 for a in ATTR_ATTRS if winners[a] == "alt")
            new_label = scores_to_label(base_score, alt_score)
            old_label = row.get("golden_label")
            if pd.isna(old_label):
                old_label = "abstain"
            else:
                old_label = str(old_label).strip().lower()
            if new_label != old_label:
                n_label_changes += 1
                df.at[idx, "golden_label"] = new_label

    df.to_parquet(GOLDEN_PATH, index=False)
    print(f"Updated {GOLDEN_PATH}")
    print(f"  Rows with phone winner changed: {n_phone_changes}")
    print(f"  Rows with golden_label changed: {n_label_changes}")


if __name__ == "__main__":
    main()
