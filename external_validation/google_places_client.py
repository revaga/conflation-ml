"""
Google Maps Places API (New) client for external validation.
Text Search (New) -> place_id -> Place Details (New) for phone, website, address, types.
Uses env GOOGLE_PLACES_API_KEY or api_keys.env; caches by (name, address) and place_id.
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KEYS_FILE = _REPO_ROOT / "api_keys.env"

# Optional disk cache (same pattern as scripts/validator_cache)
_CACHE_DIR = _REPO_ROOT / ".cache" / "google_places"
_CACHE: Optional[Any] = None

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACE_DETAILS_BASE = "https://places.googleapis.com/v1/places"
# Field masks: minimal for search; details for phone, website, address, category
SEARCH_FIELDS = "places.id"
DETAILS_FIELDS = "id,displayName,formattedAddress,types,internationalPhoneNumber,websiteUri"

# Throttle (free tier)
REQUEST_DELAY_SEC = 1.0


def _load_api_key() -> str:
    key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if key:
        return key
    if _KEYS_FILE.exists():
        with open(_KEYS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_PLACES_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def _get_cache():
    global _CACHE
    if _CACHE is None:
        try:
            import diskcache
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _CACHE = diskcache.Cache(str(_CACHE_DIR))
        except Exception as e:
            logger.warning("diskcache not available for Google Places: %s", e)
            _CACHE = {}
    return _CACHE


def _cache_key(kind: str, *parts: str) -> str:
    import hashlib
    s = f"{kind}:" + ":".join((p or "").strip().lower() for p in parts)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def search_place(name: str, address: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Text Search (New). Returns first candidate place_id or None.
    """
    key = api_key or _load_api_key()
    if not key:
        logger.warning("GOOGLE_PLACES_API_KEY not set; skipping Google Places.")
        return None
    cache = _get_cache()
    ck = _cache_key("search", name, address)
    if isinstance(cache, dict) and ck in cache:
        return cache[ck]
    try:
        time.sleep(REQUEST_DELAY_SEC)
        query = f"{name} {address}".strip()
        if not query:
            return None
        resp = requests.post(
            TEXT_SEARCH_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": SEARCH_FIELDS,
            },
            json={"textQuery": query},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        places = data.get("places") or []
        place_id = None
        if places and isinstance(places[0], dict):
            place_id = places[0].get("id")
        if place_id and hasattr(cache, "__setitem__"):
            try:
                cache[ck] = place_id
            except Exception:
                pass
        return place_id
    except requests.RequestException as e:
        logger.warning("Google Text Search failed for %s %s: %s", name, address, e)
        return None


def get_place_details(place_id: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Place Details (New). Returns dict with keys: phone, web, address, category.
    Maps Google types to a single category string (first type or primary).
    """
    key = api_key or _load_api_key()
    if not key:
        return None
    cache = _get_cache()
    ck = _cache_key("details", place_id)
    if hasattr(cache, "get") and ck in cache:
        return cache[ck]
    try:
        time.sleep(REQUEST_DELAY_SEC)
        url = f"{PLACE_DETAILS_BASE}/{place_id}"
        resp = requests.get(
            url,
            headers={
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": DETAILS_FIELDS,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        out = _parse_place_details(data)
        if out and hasattr(cache, "__setitem__"):
            try:
                cache[ck] = out
            except Exception:
                pass
        return out
    except requests.RequestException as e:
        logger.warning("Google Place Details failed for %s: %s", place_id, e)
        return None


def _parse_place_details(data: Dict[str, Any]) -> Dict[str, str]:
    """Map Places API response to our keys: phone, web, address, category."""
    out = {"phone": "", "web": "", "address": "", "category": ""}
    # internationalPhoneNumber or nationalPhoneNumber
    out["phone"] = (data.get("internationalPhoneNumber") or data.get("nationalPhoneNumber") or "").strip()
    out["web"] = (data.get("websiteUri") or "").strip()
    out["address"] = (data.get("formattedAddress") or "").strip()
    types = data.get("types")
    if isinstance(types, list) and types:
        # Use first type; optionally map to Overture-style (e.g. restaurant -> restaurant)
        out["category"] = str(types[0]).strip()
    return out


def fetch_real_data(name: str, address: str, api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience: search by name + address then get details. Returns dict with
    phone, web, address, category (empty strings when missing).
    """
    place_id = search_place(name, address, api_key)
    if not place_id:
        return {"phone": "", "web": "", "address": "", "category": ""}
    details = get_place_details(place_id, api_key)
    if not details:
        return {"phone": "", "web": "", "address": "", "category": ""}
    return details
