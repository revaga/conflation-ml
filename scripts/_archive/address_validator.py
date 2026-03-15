import re

try:
    from deepparse.parser import AddressParser  # type: ignore[import-untyped]
    address_parser = AddressParser(model_type="fasttext", device="cpu")
    _DEEPPARSE_AVAILABLE = True
except ImportError:
    address_parser = None
    _DEEPPARSE_AVAILABLE = False

def _score_fallback(address_str: str) -> int:
    """Simple heuristic when deepparse is not installed."""
    s = str(address_str).strip()
    if len(s) < 5:
        return 0
    score = 0
    if re.search(r'\d+', s):
        score += 1  # has number
    if re.search(r'\b(?:st|street|ave|avenu|road|rd|blvd|dr)\b', s, re.I):
        score += 1  # street-like
    if re.search(r'\b[A-Z]{2}\b|\b\d{5}(-\d{4})?\b', s):
        score += 1  # state or zip
    return min(score, 5)


def get_address_score(address_str):
    """Parses address using DeepParse (or fallback) and returns a completeness score."""
    if not address_str or len(str(address_str)) < 5:
        return 0

    if not _DEEPPARSE_AVAILABLE or address_parser is None:
        return _score_fallback(address_str)

    try:
        parsed_address = address_parser(address_str)
        components = parsed_address.to_dict()

        score = 0
        if components.get('StreetNumber'):
            score += 1
        if components.get('StreetName'):
            score += 1
        if components.get('Municipality'):
            score += 1
        if components.get('Province'):
            score += 1

        postcode = components.get('PostalCode')
        if postcode and re.match(r'^[A-Z0-9 -]{3,10}$', str(postcode).upper()):
            score += 1

        return score
    except Exception:
        return _score_fallback(address_str)

def compare_addresses(base_addr, alt_addr):
    """Determines which address wins the +1 point for your Step 2 logic."""
    base_score = get_address_score(base_addr)
    alt_score = get_address_score(alt_addr)
    
    if base_score > alt_score:
        return "base"
    elif alt_score > base_score:
        return "alt"
    else:
        return None