import diskcache as dc
import os
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"

# Initialize cache
cache = dc.Cache(str(CACHE_DIR / "validator_cache"))

def get_cache_key(type_str, value):
    """Generate a cache key for a specific validator and value."""
    key_str = f"{type_str}:{value}"
    return hashlib.md5(key_str.encode()).hexdigest()

def cached_validate(type_str, value, validator_func):
    """
    Check cache for validation result, otherwise run validator and store.
    type_str: 'website' or 'phone'
    """
    if not value or str(value).strip() == "":
        return False, "Empty value"
    
    key = get_cache_key(type_str, value)
    
    # Check cache
    if key in cache:
        logger.debug(f"Cache hit for {type_str}: {value}")
        return cache[key]
    
    # Run validator
    logger.debug(f"Cache miss for {type_str}: {value}. Running validator.")
    result = validator_func(value)
    
    # Store in cache
    cache[key] = result
    return result

def clear_validator_cache():
    cache.clear()
    logger.info("Validator cache cleared.")
