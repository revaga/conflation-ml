"""
Scrape a business website for phone, website, address, category using BeautifulSoup.
Uses scripts.website_validator for safe GET; caches by URL (diskcache).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
from bs4 import BeautifulSoup

TIMEOUT = 10
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u


def _fetch_url(url: str):
    """GET URL; returns (response or None, error_message)."""
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=TIMEOUT, allow_redirects=True)
        return r, None
    except requests.exceptions.RequestException as e:
        return None, str(e)

# Phone: optional +, digits, spaces/dots/dashes, at least 7 digits
_PHONE_RE = re.compile(r"(?:\+?1?[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
# Schema.org type or meta
_SCHEMA_TYPE_RE = re.compile(r'"@type"\s*:\s*"([^"]+)"', re.I)


def _get_cache():
    try:
        import diskcache
        cache_dir = _REPO_ROOT / ".cache" / "scrape_place"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return diskcache.Cache(str(cache_dir))
    except Exception:
        return {}


def _cache_key(url: str) -> str:
    import hashlib
    return hashlib.md5(url.strip().lower().encode("utf-8")).hexdigest()


def _extract_phones(html: str) -> str:
    """First phone number found in text (strip to digits for consistency)."""
    matches = _PHONE_RE.findall(html)
    if not matches:
        return ""
    raw = matches[0]
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 7:
        return raw.strip()
    return ""


def _extract_address_schema(soup: BeautifulSoup) -> str:
    """Try schema.org PostalAddress or streetAddress."""
    for tag in soup.find_all(string=re.compile(r"streetAddress|addressLocality|postalCode")):
        parent = tag.parent if hasattr(tag, "parent") else None
        if parent and parent.name:
            text = parent.get_text(separator=" ", strip=True)
            if len(text) > 10:
                return text[:200]
    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                addr = data.get("address") or data.get("streetAddress")
                if isinstance(addr, dict):
                    parts = [addr.get("streetAddress"), addr.get("addressLocality"), addr.get("addressRegion"), addr.get("postalCode")]
                    return " ".join(p for p in parts if p)
                if isinstance(addr, str):
                    return addr
        except Exception:
            pass
    return ""


def _extract_category_schema(soup: BeautifulSoup) -> str:
    """Schema.org @type like LocalBusiness, Restaurant."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                t = data.get("@type")
                if isinstance(t, str) and t not in ("WebPage", "Organization"):
                    return t
                if isinstance(t, list):
                    for x in t:
                        if isinstance(x, str) and x not in ("WebPage", "Organization"):
                            return x
        except Exception:
            pass
    return ""


def scrape_place(url: str, place_name: Optional[str] = None, use_cache: bool = True) -> Dict[str, str]:
    """
    Fetch URL and extract phone, website, address, category.
    Returns dict with keys phone, web, address, category (empty string when not found).
    """
    url = _normalize_url(url)
    if not url:
        return {"phone": "", "web": "", "address": "", "category": ""}
    cache = _get_cache()
    if use_cache and hasattr(cache, "get"):
        ck = _cache_key(url)
        if ck in cache:
            return cache[ck]
    resp, err = _fetch_url(url)
    if resp is None or resp.status_code >= 400:
        return {"phone": "", "web": "", "address": "", "category": ""}
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        phone = _extract_phones(text)
        address = _extract_address_schema(soup)
        if not address and len(text) > 20:
            # Fallback: look for lines that look like addresses (number + street word)
            for line in text.split("\n"):
                line = line.strip()
                if re.search(r"\d+\s+\w+\s+(st|street|ave|avenue|blvd|road|rd|dr)", line, re.I) and len(line) < 150:
                    address = line
                    break
        category = _extract_category_schema(soup)
        # Website: canonical or given URL (normalized)
        web = url
        link = soup.find("link", rel="canonical")
        if link and link.get("href"):
            web = link["href"].strip()
        if not web.startswith("http"):
            web = "https://" + web
        out = {
            "phone": phone,
            "web": re.sub(r"^https?://", "", web).strip("/").lower(),
            "address": address,
            "category": category,
        }
        if use_cache and hasattr(cache, "__setitem__"):
            try:
                cache[_cache_key(url)] = out
            except Exception:
                pass
        return out
    except Exception:
        return {"phone": "", "web": "", "address": "", "category": ""}
