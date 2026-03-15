import json
import re
from typing import Any

import numpy as np
import pandas as pd


def _expand_abbreviations(address: str) -> str:
    if not address or pd.isna(address):
        return ""

    abbr_map = {
        r"\bst\b": "street",
        r"\bave\b": "avenue",
        r"\bdr\b": "drive",
        r"\brd\b": "road",
        r"\bblvd\b": "boulevard",
        r"\bln\b": "lane",
        r"\bct\b": "court",
        r"\bpl\b": "place",
        r"\bsq\b": "square",
        r"\bpkwy\b": "parkway",
        r"\bcir\b": "circle",
        r"\bhwy\b": "highway",
    }

    normalized = str(address).lower()
    for pattern, replacement in abbr_map.items():
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def normalize_address_json(val: Any) -> str:
    """
    Normalize a JSON-encoded address field into a canonical freeform string with
    abbreviations expanded. This mirrors the Phase 1 behavior.
    """
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            if isinstance(data, dict):
                freeform = data.get("freeform", "")
            else:
                freeform = str(val)
        except Exception:
            freeform = str(val)
    elif isinstance(val, dict):
        freeform = val.get("freeform", "")
    else:
        freeform = ""

    return _expand_abbreviations(freeform)


def standardize_phone(phone: Any) -> str:
    """Strip non-digits and normalize missing phones to empty string."""
    if phone is None or (isinstance(phone, float) and np.isnan(phone)):
        return ""
    if not phone:
        return ""
    return re.sub(r"\D", "", str(phone))


def normalize_website(url: Any) -> str:
    """Normalize website to bare domain, consistent with Phase 1 logic."""
    if url is None or (isinstance(url, float) and np.isnan(url)):
        return ""
    if not url:
        return ""

    url = str(url).lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.strip("/")


def process_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Phase-1 style normalized address columns in-place:
    - norm_conflated_addr
    - norm_base_addr
    """
    df = df.copy()
    df["norm_conflated_addr"] = df["addresses"].apply(normalize_address_json)
    df["norm_base_addr"] = df["base_addresses"].apply(normalize_address_json)
    return df

