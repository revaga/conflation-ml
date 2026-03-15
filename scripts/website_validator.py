import re
import requests
import logging
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TIMEOUT = 10

# Match http(s) URL; capture until whitespace, quote, bracket, or end (handles wrapped URLs from JSON)
_URL_PATTERN = re.compile(r"(https?://[^\s\]\"\'\}\>,]+)", re.IGNORECASE)

def _extract_url(raw):
    """Extract a single http(s) URL from a string that may be wrapped."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # If it already looks like a plain URL, use it
    s = s.strip("[]\"'{}")
    match = _URL_PATTERN.search(s)
    if match:
        return match.group(1).rstrip(".,;:)")
    if s.startswith(("http://", "https://")):
        return s
    return s

def _normalize_url(url):
    """Ensure URL has a scheme."""
    u = _extract_url(url)
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
    except requests.exceptions.Timeout:
        return None, "Timeout"
    except requests.exceptions.ConnectionError:
        return None, "Connection Error"
    except requests.exceptions.RequestException as e:
        return None, str(e)

def _name_words(name):
    """Extract words of length > 2 from a place name."""
    if not name or not isinstance(name, str):
        return []
    text = re.sub(r"[^\w\s]", " ", name)
    return [w for w in text.split() if len(w) > 2]

def _page_contains_name(html_text, place_name):
    """Return True if any significant word from place_name appears in html_text."""
    words = _name_words(place_name)
    if not words:
        return True
    text_lower = (html_text or "").lower()
    return any(w.lower() in text_lower for w in words)

def verify_website(url, place_name=None):
    """Verify that a website is reachable and optionally that the place name appears."""
    if url is not None and not isinstance(url, str):
        url = str(url).strip()
    url = _normalize_url(url)
    if not url:
        return False, "Empty URL"

    try:
        parsed = urlparse(url)
        schemes_to_try = [parsed.scheme]
        if parsed.scheme == "https":
            schemes_to_try.append("http")
        else:
            schemes_to_try.append("https")
        urls_to_try = [
            urlunparse((s, parsed.netloc, parsed.path or "/", "", parsed.query, parsed.fragment))
            for s in schemes_to_try
        ]
    except ValueError:
        urls_to_try = [url]
        if url.startswith("https://"):
            urls_to_try.append("http://" + url[8:])
        elif url.startswith("http://"):
            urls_to_try.append("https://" + url[7:])

    response = None
    last_error = None
    for try_url in urls_to_try:
        logger.debug(f"Verifying URL: {try_url}")
        resp, err = _fetch_url(try_url)
        if resp is not None and resp.status_code < 400:
            response = resp
            last_error = None
            break
        if resp is not None:
            last_error = f"Status: {resp.status_code}"
        else:
            last_error = err

    if response is None:
        return False, last_error or "Request failed"

    if place_name and response.headers.get("content-type", "").lower().find("html") >= 0:
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            body_text = soup.get_text(separator=" ", strip=True)
            if not _page_contains_name(body_text, place_name):
                return False, "Place name not found on page"
        except Exception:
            pass

    return True, f"Status: {response.status_code}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_urls = [
        "https://www.google.com",
        "http://www.goinpostaljacksonville.com/",
        "https://thiswebsiteisdefinitelyfake12345.com",
    ]
    for url in test_urls:
        exists, details = verify_website(url)
        logger.info(f"URL: {url} | Valid: {exists} | Details: {details}")
