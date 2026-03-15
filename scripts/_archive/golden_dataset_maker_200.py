"""
Golden dataset maker for 200 records.
Loads labeled data from golden_dataset_100.parquet (first 100) and lets you
label records 101-200. Outputs to golden_dataset_200.parquet.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from parquet_io import read_parquet_safe
from phonenumber_validator import validate_phone_number, try_with_region
from website_validator import verify_website

# Paths when run as script (project root = parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
PHASE1_PATH = PROJECT_ROOT / "data" / "phase1_processed.parquet"
GOLDEN_100_PATH = PROJECT_ROOT / "data" / "golden_dataset_100.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "golden_dataset_200.parquet"
NUM_RECORDS = 200
SAVE_EVERY_N = 5
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
# Bookkeeping columns included in the golden dataset output (base/alt/both/none per attribute)
BOOKKEEPING_COLUMNS = ["golden_label"] + [f"attr_{a}_winner" for a in ATTR_ATTRS]
# Phone numbers corrected with country code from address (when validation failed without it)
PHONE_CORRECTED_COLUMNS = ["base_phone_with_country", "alt_phone_with_country"]


def _prompt_superiority(label, base_display, alt_display):
    """
    Ask human which value is better. Returns 'base', 'alt', 'both', or None (skip).
    """
    print(f"\n--- {label} ---")
    print(f"  base:  {base_display}")
    print(f"  alt:  {alt_display}")
    while True:
        choice = input("Which is better? (base / alt / both / none): ").strip().lower()
        if choice in ("base", "b"):
            return "base"
        if choice in ("match", "m", "alt", "a"):
            return "alt"
        if choice in ("both", "t"):
            return "both"
        if choice in ("none", "n", "skip", "s", ""):
            return None
        print("Please enter: base, alt, both, or none")


def _winner_str(w):
    """Normalize winner to 'base', 'alt', 'both', or 'none' for bookkeeping."""
    return w if w in ("base", "alt", "both") else "none"


def generate_golden_label(row):
    """
    Golden label per spec: base / alt / abstain from comparing core attributes.
    Every attribute is recorded as one of: base, alt, both, none.
    Name, address, category: human choice (base / alt / both / none).
    Phone, website: automatic (both valid → both, neither valid → none, else base or alt).
    +2 margin rule; special edge case when one has 0 valid and the other has >= 2.
    Returns (label, bookkeeping_dict, corrected_phones_dict).
    corrected_phones_dict has base_phone_with_country, alt_phone_with_country when we fixed with country code.
    """
    # STEP 1 — Initialize scores and bookkeeping
    base_score = 0
    alt_score = 0
    bookkeeping = {}
    corrected_phones = {}

    def _region_from_address(addr):
        """Infer country/region code from address text for phone parsing (e.g. US, CA, GB)."""
        if addr is None or (isinstance(addr, float) and pd.isna(addr)):
            return None
        text = str(addr).upper()
        # Explicit country mentions
        if " UNITED STATES" in text or " USA" in text or " U.S." in text or " U.S.A" in text:
            return "US"
        if " CANADA" in text or " CAN" in text or " ON " in text or " BC " in text or " AB " in text or " QB " in text:
            return "CA"
        if " UNITED KINGDOM" in text or " UK" in text or " GB " in text or " ENGLAND" in text or " SCOTLAND" in text:
            return "GB"
        # US state abbreviations (two-letter before ZIP pattern or standalone)
        if re.search(r"\b(AK|AL|AR|AZ|CA|CO|CT|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)\b", text):
            return "US"
        return None

    # STEP 2 — Evaluate each core attribute (phones, websites, addresses, categories, name)
    # Place name — human selection (or auto "both" if same)
    def _name_display(val):
        if val is None or (isinstance(val, float) and str(val) == 'nan'):
            return "(empty)"
        if isinstance(val, dict):
            return (val.get('primary') or val.get('raw') or str(val)).strip() or "(empty)"
        return str(val).strip() or "(empty)"

    base_name_display = _name_display(row.get('base_name') or row.get('base_names'))
    alt_name_display = _name_display(row.get('alt_name') or row.get('names'))
    if base_name_display == alt_name_display:
        print("\n--- Place name ---")
        print(f"  base:  {base_name_display}")
        print(f"  alt:  {alt_name_display}")
        print("  Same → both")
        name_winner = "both"
    else:
        name_winner = _prompt_superiority("Place name", base_name_display, alt_name_display)
    bookkeeping["name"] = _winner_str(name_winner)
    if name_winner == 'base':
        base_score += 1
    elif name_winner == 'alt':
        alt_score += 1

    # 📞 Phones — 1) validate with phonenumber_validator; 2) if invalid, try country code; 3) both valid→both, none→none, one valid→that one
    def _first_str(val):
        if isinstance(val, list) and len(val) > 0:
            return str(val[0]).strip() or None
        return val

    def _phone_str(val):
        """Coerce to string for phone processing; None/NaN/empty -> None."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        return s if s and s.lower() not in ("nan", "none") else None

    base_phone = _phone_str(_first_str(row.get('base_phone') or row.get('norm_base_phone') or row.get('base_phones')))
    alt_phone = _phone_str(_first_str(row.get('alt_phone') or row.get('norm_conflated_phone') or row.get('phones')))
    base_addr_for_region = row.get('base_address') or row.get('norm_base_addr') or row.get('base_addresses')
    alt_addr_for_region = row.get('alt_address') or row.get('norm_conflated_addr') or row.get('addresses')
    base_region = _region_from_address(base_addr_for_region) or "US"
    alt_region = _region_from_address(alt_addr_for_region) or "US"

    # Step 1: validate with phonenumber_validator
    base_valid = validate_phone_number(base_phone)[0] if base_phone else False
    alt_valid = validate_phone_number(alt_phone)[0] if alt_phone else False
    # Step 2: if invalid, try country code; store corrected E164 and treat as valid for decision
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
    # Step 3: both valid → both, none valid → none, only one valid → that one
    if base_valid and alt_valid:
        phone_winner = "both"
    elif not base_valid and not alt_valid:
        phone_winner = None  # → "none"
    elif base_valid:
        phone_winner = "base"
    else:
        phone_winner = "alt"
    _phone_show = lambda v: "(empty)" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    print("\n--- Phone ---")
    print(f"  base:  {_phone_show(base_phone)}")
    print(f"  alt:  {_phone_show(alt_phone)}")
    print(f"  → {_winner_str(phone_winner)}")
    bookkeeping["phone"] = _winner_str(phone_winner)
    if phone_winner == 'base':
        base_score += 1
    elif phone_winner == 'alt':
        alt_score += 1

    # 🌐 Websites — auto when one or both validate; if neither validates, human chooses (no URL preprocessing)
    def check_web_superiority(base_web, alt_web):
        base_url = None if not base_web or (isinstance(base_web, float) and pd.isna(base_web)) else str(base_web).strip() or None
        alt_url = None if not alt_web or (isinstance(alt_web, float) and pd.isna(alt_web)) else str(alt_web).strip() or None
        base_valid, _ = verify_website(base_url) if base_url else (False, None)
        alt_valid, _ = verify_website(alt_url) if alt_url else (False, None)
        if base_valid and not alt_valid:
            return "base"
        if not base_valid and alt_valid:
            return "alt"
        if base_valid and alt_valid:
            return "both"
        return None  # neither validated → human will choose

    # Column fallbacks: phase1 = norm_base_website / norm_conflated_website; raw = base_websites / websites
    base_web = _first_str(row.get('base_web') or row.get('norm_base_website') or row.get('base_websites'))
    alt_web = _first_str(row.get('alt_web') or row.get('norm_conflated_website') or row.get('websites'))
    web_winner = check_web_superiority(base_web, alt_web)
    _web_show = lambda v: "(empty)" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    if web_winner is None:
        print("\n  (Website: neither validated automatically — please choose)")
        web_winner = _prompt_superiority("Website", _web_show(base_web), _web_show(alt_web))
    else:
        print("\n--- Website ---")
        print(f"  base:  {_web_show(base_web)}")
        print(f"  alt:  {_web_show(alt_web)}")
        print(f"  → {_winner_str(web_winner)}")
    bookkeeping["web"] = _winner_str(web_winner)
    if web_winner == 'base':
        base_score += 1
    elif web_winner == 'alt':
        alt_score += 1

    # 🏠 Addresses — auto "both" if exact same (and not empty); if same but one has more items in array, that one wins; else human
    base_addr_raw = row.get('base_address') or row.get('norm_base_addr') or row.get('base_addresses')
    alt_addr_raw = row.get('alt_address') or row.get('norm_conflated_addr') or row.get('addresses')

    def _addr_show(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "(empty)"
        return str(val)

    def _addr_to_list(val):
        """Normalize address to list of non-empty strings for comparison."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, dict):
            s = (val.get("freeform") or val.get("raw") or str(val)).strip()
            return [s] if s else []
        s = str(val).strip()
        return [s] if s else []

    def _addr_completeness(val):
        """Count non-empty fields in address (for list of dicts: e.g. region empty = less complete)."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0
        if isinstance(val, list):
            total = 0
            for item in val:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if v is not None and str(v).strip():
                            total += 1
                else:
                    if str(item).strip():
                        total += 1
            return total
        if isinstance(val, dict):
            return sum(1 for k, v in val.items() if v is not None and str(v).strip())
        return 1 if str(val).strip() else 0

    base_list = _addr_to_list(base_addr_raw)
    alt_list = _addr_to_list(alt_addr_raw)
    addr_winner = None
    if base_list or alt_list:
        if set(base_list) == set(alt_list):
            print("\n--- Address ---")
            print(f"  base:  {_addr_show(base_addr_raw)}")
            print(f"  alt:  {_addr_show(alt_addr_raw)}")
            print("  Same (any order) → both")
            addr_winner = "both"
        elif len(base_list) > len(alt_list) and set(alt_list) <= set(base_list):
            print("\n--- Address ---")
            print(f"  base:  {_addr_show(base_addr_raw)}")
            print(f"  alt:  {_addr_show(alt_addr_raw)}")
            print("  Base has more items → base")
            addr_winner = "base"
        elif len(alt_list) > len(base_list) and set(base_list) <= set(alt_list):
            print("\n--- Address ---")
            print(f"  base:  {_addr_show(base_addr_raw)}")
            print(f"  alt:  {_addr_show(alt_addr_raw)}")
            print("  Alt has more items → alt")
            addr_winner = "alt"
        if addr_winner is None:
            base_complete = _addr_completeness(base_addr_raw)
            alt_complete = _addr_completeness(alt_addr_raw)
            if base_complete > alt_complete:
                print("\n--- Address ---")
                print(f"  base:  {_addr_show(base_addr_raw)}")
                print(f"  alt:  {_addr_show(alt_addr_raw)}")
                print("  Base has fewer empty fields → base")
                addr_winner = "base"
            elif alt_complete > base_complete:
                print("\n--- Address ---")
                print(f"  base:  {_addr_show(base_addr_raw)}")
                print(f"  alt:  {_addr_show(alt_addr_raw)}")
                print("  Alt has fewer empty fields → alt")
                addr_winner = "alt"
    if addr_winner is None:
        addr_winner = _prompt_superiority(
            "Address",
            _addr_show(base_addr_raw),
            _addr_show(alt_addr_raw),
        )
    bookkeeping["address"] = _winner_str(addr_winner)
    if addr_winner == 'base':
        base_score += 1
    elif addr_winner == 'alt':
        alt_score += 1

    # 🗂 Categories — human selection (or auto "both" if same)
    def _category_display(val):
        if val is None or (isinstance(val, float) and str(val) == 'nan'):
            return "(empty)"
        if isinstance(val, dict):
            s = (val.get('primary') or val.get('raw') or "").strip()
            return s or "(empty)"
        return str(val).strip() or "(empty)"

    base_cat_display = _category_display(row.get('base_categories'))
    alt_cat_display = _category_display(row.get('alt_categories') or row.get('categories'))
    if base_cat_display == alt_cat_display:
        print("\n--- Category ---")
        print(f"  base:  {base_cat_display}")
        print(f"  alt:  {alt_cat_display}")
        print("  Same → both")
        cat_winner = "both"
    else:
        cat_winner = _prompt_superiority("Category", base_cat_display, alt_cat_display)
    bookkeeping["category"] = _winner_str(cat_winner)
    if cat_winner == 'base':
        base_score += 1
    elif cat_winner == 'alt':
        alt_score += 1

    # STEP 3 — Compare total scores
    # Special edge case: one has 0 valid, the other has >= 2 → choose the one with valid
    if base_score == 0 and alt_score >= 2:
        return "alt", bookkeeping, corrected_phones
    if alt_score == 0 and base_score >= 2:
        return "base", bookkeeping, corrected_phones
    # +2 margin rule
    if base_score >= alt_score + 2:
        return "base", bookkeeping, corrected_phones
    if alt_score >= base_score + 2:
        return "alt", bookkeeping, corrected_phones
    return "abstain", bookkeeping, corrected_phones


def _save_golden(df: pd.DataFrame, path: Path) -> None:
    """Write golden dataset with all bookkeeping and phone-corrected columns."""
    other_cols = [c for c in df.columns if c not in BOOKKEEPING_COLUMNS and c not in PHONE_CORRECTED_COLUMNS]
    ordered = other_cols + [c for c in PHONE_CORRECTED_COLUMNS if c in df.columns] + [c for c in BOOKKEEPING_COLUMNS if c in df.columns]
    df[ordered].to_parquet(path, index=False)


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    df = read_parquet_safe(str(DATA_PATH))
    subset = df.head(NUM_RECORDS).copy()

    # Add missing columns from project_a_samples (subset is from it; ensures we have all columns if source ever changes)
    samples_extra = [c for c in df.columns if c not in subset.columns]
    if samples_extra:
        samples_by_id = df.set_index("id")
        for c in samples_extra:
            subset[c] = subset["id"].map(samples_by_id[c])

    # Add missing columns from phase1_processed (norm_*, *_similarity, etc.) so output has them
    if PHASE1_PATH.exists():
        phase1 = read_parquet_safe(str(PHASE1_PATH))
        extra_cols = [c for c in phase1.columns if c not in subset.columns]
        if extra_cols:
            phase1_by_id = phase1.set_index("id")
            for c in extra_cols:
                subset[c] = subset["id"].map(phase1_by_id[c])

    # Initialize: prefer golden_dataset_200 for resume; else seed from golden_dataset_100
    n_done = 0
    if OUTPUT_PATH.exists():
        results_df = read_parquet_safe(str(OUTPUT_PATH))
        n_done = len(results_df)
        print(f"Resuming: {n_done} records already labeled in {OUTPUT_PATH}")
    elif GOLDEN_100_PATH.exists():
        results_df = read_parquet_safe(str(GOLDEN_100_PATH))
        n_done = len(results_df)
        print(f"Loaded {n_done} records from {GOLDEN_100_PATH}. Labeling records {n_done + 1} to {NUM_RECORDS}.")
    else:
        results_df = pd.DataFrame()
        print(f"No existing golden data found. Labeling records 1 to {NUM_RECORDS}.")

    for i in range(n_done, NUM_RECORDS):
        row = subset.iloc[i]
        rec_num = i + 1
        print(f"\n{'='*60} Record {rec_num} / {NUM_RECORDS} {'='*60}")
        label, bookkeeping, corrected_phones = generate_golden_label(row)
        print(f"  → golden_label = {label}")

        # Append this row with label, bookkeeping, and corrected phone columns when applicable
        new_row = row.to_dict()
        new_row["golden_label"] = label
        for attr in ATTR_ATTRS:
            new_row[f"attr_{attr}_winner"] = bookkeeping[attr]
        new_row["base_phone_with_country"] = corrected_phones.get("base_phone_with_country")
        new_row["alt_phone_with_country"] = corrected_phones.get("alt_phone_with_country")
        new_df = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_df], ignore_index=True)

        # Save every SAVE_EVERY_N records (ensure bookkeeping columns are present and ordered)
        if (i - n_done + 1) % SAVE_EVERY_N == 0:
            _save_golden(results_df, OUTPUT_PATH)
            print(f"  [Saved {len(results_df)} records to {OUTPUT_PATH}]")

        if i < NUM_RECORDS - 1:
            choice = input("\nPress Enter for next record, or 'pause' to save and quit: ").strip().lower()
            if choice in ("pause", "p", "quit", "q"):
                _save_golden(results_df, OUTPUT_PATH)
                print(f"Paused. Saved {len(results_df)} records to {OUTPUT_PATH}. Run again to continue.")
                break

    _save_golden(results_df, OUTPUT_PATH)
    print(f"\nDone. Saved {len(results_df)} labeled records to {OUTPUT_PATH}")
