"""
Non-Google search (DuckDuckGo) to find place info from query "name + address".
Parses snippets for phone, website, address. Optional: duckduckgo-search library.
For research/prototyping only; search result scraping may conflict with ToS.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PHONE_RE = re.compile(r"(?:\+?1?[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
_URL_RE = re.compile(r"https?://[^\s\]\"\'\}\>,]+", re.I)
# Simple address: number + street word
_ADDR_RE = re.compile(r"\d+\s+[\w\s]+(?:street|st|avenue|ave|blvd|road|rd|drive|dr|lane|ln|court|ct)\b[\w\s,]*", re.I)

MAX_RESULTS = 5


def _search_duckduckgo(query: str, max_results: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """Run DuckDuckGo text search. Returns list of {title, href, body}."""
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        return results if results else []
    except ImportError:
        return []
    except Exception:
        return []


def _extract_phone(text: str) -> str:
    m = _PHONE_RE.search(text)
    return m.group(0).strip() if m else ""


def _extract_url(text: str) -> str:
    m = _URL_RE.search(text)
    if m:
        u = m.group(0).rstrip(".,;:)")
        return u
    return ""


def _extract_address(text: str) -> str:
    m = _ADDR_RE.search(text)
    return m.group(0).strip() if m else ""


def search_place(query: str, max_results: int = MAX_RESULTS) -> Dict[str, str]:
    """
    Run non-Google search for "place name + address" and merge first results into
    a single record: phone, web, address, category (category often empty from snippets).
    """
    out = {"phone": "", "web": "", "address": "", "category": ""}
    results = _search_duckduckgo(query, max_results)
    for r in results:
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "")
        href = (r.get("href") or "")
        body = (r.get("body") or "")
        combined = f"{title} {href} {body}"
        if not out["phone"]:
            out["phone"] = _extract_phone(combined)
        if not out["web"] and href:
            out["web"] = href.strip()
        if not out["address"]:
            out["address"] = _extract_address(combined)
        if out["phone"] and out["web"] and out["address"]:
            break
    # Normalize web to bare domain for comparison
    if out["web"]:
        w = out["web"].lower()
        w = re.sub(r"^https?://", "", w)
        w = re.sub(r"^www\.", "", w)
        out["web"] = w.strip("/")
    return out
