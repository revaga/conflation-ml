import json
import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path

# Allow "from scripts.parquet_io import ..." when run as python scripts/slm_attribute_labeler.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from openai import OpenAI
from scripts.parquet_io import read_parquet_safe

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "phase1_processed.parquet"
OUTPUT_PATH = DATA_DIR / "phase3_slm_labeled.parquet"
SAVE_EVERY_N = 50
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

# API keys: read from env first, then from repo file api_keys.env (so you can set keys in file).
KEYS_FILE = PROJECT_ROOT / "api_keys.env"

def _load_keys_from_file(path: Path) -> dict:
    """Load KEY=value pairs from a file; strip whitespace, skip empty lines and # comments."""
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out

_keys_file = _load_keys_from_file(KEYS_FILE)

def _get(key: str, default: str = "") -> str:
    """Get key from env, then from api_keys.env."""
    return os.getenv(key) or _keys_file.get(key, default)

# Provider selection (can be overridden by argparse). No OpenAI API key used.
_has_hf = bool(_get("HF_TOKEN") or _get("HUGGINGFACE_API_KEY"))
_provider_default = "huggingface" if _has_hf else "local"
SLM_PROVIDER = _get("SLM_PROVIDER", _provider_default).lower()
if SLM_PROVIDER not in ("local", "huggingface", "ollama"):
    SLM_PROVIDER = "local"

HF_TOKEN = _get("HF_TOKEN") or _get("HUGGINGFACE_API_KEY")
HF_ROUTER_URL = "https://router.huggingface.co/v1"

# Ollama (OpenAI-compatible API: local server or cloud e.g. kimi-k2.5:cloud)
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = _get("OLLAMA_API_KEY", "ollama")

# Model name: use SLM_MODEL if set; otherwise default by provider
if SLM_PROVIDER == "local":
    _default_model = "google/gemma-3-1b-it"
elif SLM_PROVIDER == "huggingface":
    _default_model = "HuggingFaceH4/zephyr-7b-beta"
elif SLM_PROVIDER == "ollama":
    _default_model = "kimi-k2.5:cloud"
else:
    _default_model = "kimi-k2.5:cloud"

MODEL_NAME = _get("SLM_MODEL") or _default_model

# Match golden_dataset_maker schema
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
BOOKKEEPING_COLUMNS = ["golden_label"] + [f"attr_{a}_winner" for a in ATTR_ATTRS]

# Reasoning rules (used in basic prompt)
PROMPT_RULES = """
When comparing two records, use this reasoning:
- phone number with area code is better
- website name that matches place name is better
- address that is better formatted is better
- category that matches place name is better
- the non-empty value is better
"""

# Optimized prompt for API models (clearer structure, stricter output, fewer "none")
PROMPT_OPENAI_OPTIMIZED = """You are a data steward comparing two records for the same place: "base" (authoritative) and "alt" (conflated). For each attribute, choose exactly one: base, alt, both, or none.

Rules:
- base = base value is better or only valid
- alt = alt value is better or only valid  
- both = values are equivalent or both acceptable
- none = both empty or both unusable (use sparingly; prefer base/alt when one is clearly better)

Prefer base or alt when one value is clearly better; use "both" when equivalent. Use "none" only when both are empty or invalid.

Attribute values (base vs alt):
{attr_block}

Output: a single JSON object only. No markdown, no explanation. Use exactly these keys and values.
Example: {{"name":"base","phone":"both","web":"alt","address":"base","category":"both","reason":"One sentence."}}"""

def _winner_str(w):
    """Normalize winner to 'base', 'alt', 'both', or 'none' for bookkeeping."""
    return w if w in ("base", "alt", "both") else "none"

# --- Display helpers ---
def _name_display(val):
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return "(empty)"
    if isinstance(val, dict):
        return (val.get("primary") or val.get("raw") or str(val)).strip() or "(empty)"
    return str(val).strip() or "(empty)"

def _first_str(val):
    if isinstance(val, list) and len(val) > 0:
        return str(val[0]).strip() or None
    return val

def _phone_show(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "(empty)"
    return str(val)

def _addr_show(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "(empty)"
    return str(val)

def _category_display(val):
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return "(empty)"
    if isinstance(val, dict):
        s = (val.get("primary") or val.get("raw") or "").strip()
        return s or "(empty)"
    return str(val).strip() or "(empty)"

def get_display_values(row):
    """Return dict of base/alt display strings for each attribute (for prompt)."""
    base_name = _name_display(row.get("base_names"))
    alt_name = _name_display(row.get("names"))
    base_phone = _phone_show(_first_str(row.get("norm_base_phone") or row.get("base_phones")))
    alt_phone = _phone_show(_first_str(row.get("norm_conflated_phone") or row.get("phones")))
    base_web = _phone_show(_first_str(row.get("norm_base_website") or row.get("base_websites")))
    alt_web = _phone_show(_first_str(row.get("norm_conflated_website") or row.get("websites")))
    base_addr = _addr_show(row.get("norm_base_addr") or row.get("base_addresses"))
    alt_addr = _addr_show(row.get("norm_conflated_addr") or row.get("addresses"))
    base_cat = _category_display(row.get("base_categories"))
    alt_cat = _category_display(row.get("categories"))
    return {
        "name": (base_name, alt_name),
        "phone": (base_phone, alt_phone),
        "web": (base_web, alt_web),
        "address": (base_addr, alt_addr),
        "category": (base_cat, alt_cat),
    }

def construct_prompt(row):
    """Build a prompt with rules reasoning, then per-attribute base/alt values and classification ask."""
    displays = get_display_values(row)

    blocks = []
    for attr in ATTR_ATTRS:
        base_val, alt_val = displays[attr]
        blocks.append(f"  {attr}:  base = {base_val!r}   alt = {alt_val!r}")
    attr_block = "\n".join(blocks)

    return f"""{PROMPT_RULES}

You are a data steward. We have two sources for one place: "base" and "alt" (conflated). For each attribute below, choose the BETTER SOURCE: base, alt, both, or none.
- base = base record's value is better or only valid
- alt = alt record's value is better or only valid
- both = equivalent or both acceptable
- none = neither is useful (both empty/invalid)

Attribute values (base vs alt):
{attr_block}

Respond with ONLY a JSON object (no markdown). Use exactly these keys and values in {{"base","alt","both","none"}} per attribute. Optionally include "golden_label" (one of: base, alt, abstain) and "reason" (short sentence).
Example: {{"name":"both","phone":"base","web":"alt","address":"base","category":"both","golden_label":"base","reason":"..."}}"""


def construct_prompt_optimized(row):
    """Optimized prompt for API models: clearer structure, fewer 'none' outputs."""
    displays = get_display_values(row)
    blocks = []
    for attr in ATTR_ATTRS:
        base_val, alt_val = displays[attr]
        blocks.append(f"  {attr}:  base = {base_val!r}   alt = {alt_val!r}")
    attr_block = "\n".join(blocks)
    return PROMPT_OPENAI_OPTIMIZED.format(attr_block=attr_block)


def get_client():
    """Return a client object appropriate for the selected provider."""
    if SLM_PROVIDER == "local":
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            logger.error("For provider=local install transformers and torch: pip install transformers torch")
            return None
        hf_token = _get("HF_TOKEN") or _get("HUGGINGFACE_API_KEY") or None
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        return {"tokenizer": tokenizer, "model": model}

    if SLM_PROVIDER == "huggingface":
        if not HF_TOKEN:
            logger.warning("No Hugging Face token. Set HF_TOKEN or HUGGINGFACE_API_KEY in env or api_keys.env.")
            return None
        return OpenAI(api_key=HF_TOKEN, base_url=HF_ROUTER_URL)

    if SLM_PROVIDER == "ollama":
        return OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

    return None

def _extract_json(text: str):
    """Try to parse JSON from model output."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON found in model output")

def call_slm(client, prompt, model=None):
    """Call SLM; return parsed JSON or empty dict on failure."""
    if model is None:
        model = MODEL_NAME
    if not client:
        return {
            "name": "none", "phone": "none", "web": "none", "address": "none", "category": "none",
            "golden_label": "abstain", "reason": "API client missing."
        }
    try:
        if SLM_PROVIDER == "local":
            tokenizer = client["tokenizer"]
            model_obj = client["model"]
            full_prompt = (
                "You output only valid JSON with keys: name, phone, web, address, category "
                "(each value: base, alt, both, or none). Optionally golden_label (base/alt/abstain) and reason.\n\n"
                + prompt
            )
            messages = [{"role": "user", "content": full_prompt}]
            try:
                prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                prompt_str = full_prompt
            
            encoded = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=8192)
            input_ids = encoded["input_ids"].to(model_obj.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_obj.device)
            else:
                import torch
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
            
            gen_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "max_new_tokens": 512}
            pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
                
            outputs = model_obj.generate(**gen_kwargs)
            content = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            return _extract_json(content)
            
        # Cloud / API providers (openai, ollama, huggingface)
        system = "You output only valid JSON with keys: name, phone, web, address, category. Each value must be one of: base, alt, both, none. No markdown."
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return _extract_json(content)
    except Exception as e:
        raise RuntimeError(str(e))

def call_slm_with_retries(client, prompt, model=None):
    """Call SLM with exponential backoff."""
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            return call_slm(client, prompt, model=model or MODEL_NAME)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"reason": f"Error: {e}"}
            time.sleep(backoff)
            backoff *= 2
    return {}

def parse_slm_response(raw):
    """Normalize SLM response."""
    result = {}
    for attr in ATTR_ATTRS:
        val = str(raw.get(attr) or "none").strip().lower()
        result[f"attr_{attr}_winner"] = _winner_str(val)
    result["golden_label"] = (raw.get("golden_label") or "abstain").strip().lower()
    if result["golden_label"] not in ("base", "alt", "abstain"):
        result["golden_label"] = "abstain"
    result["SLM_reason"] = raw.get("reason") or ""
    return result

def verify_model_responds(client, sample_row=None) -> bool:
    """Verify the model is functioning."""
    if client is None:
        logger.error("Verification failed: no API client.")
        return False
    if sample_row is not None:
        prompt = construct_prompt(sample_row)
    else:
        prompt = "Test response with JSON."
    try:
        raw = call_slm(client, prompt)
        parsed = parse_slm_response(raw)
        logger.info(f"Verification OK: model responded (provider={SLM_PROVIDER}, model={MODEL_NAME})")
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def _save_results(df: pd.DataFrame, path: Path) -> None:
    """Write results to parquet."""
    other = [c for c in df.columns if c not in BOOKKEEPING_COLUMNS and c != "SLM_reason"]
    ordered = other + [c for c in BOOKKEEPING_COLUMNS if c in df.columns]
    if "SLM_reason" in df.columns:
        ordered.append("SLM_reason")
    df[ordered].to_parquet(path, index=False)

def main():
    global SLM_PROVIDER, MODEL_NAME, OUTPUT_PATH
    parser = argparse.ArgumentParser(description="Label phase1 records with SLM attribute winners.")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path (default: data/phase3_slm_labeled.parquet)")
    parser.add_argument("--provider", type=str, default=None, choices=["local", "huggingface", "ollama"], help="SLM provider (overrides env)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. gpt-4o-mini)")
    args = parser.parse_args()
    if args.output is not None:
        OUTPUT_PATH = Path(args.output)
        if not OUTPUT_PATH.is_absolute():
            OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
    if args.provider is not None:
        SLM_PROVIDER = args.provider
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        # When provider is overridden, use that provider's default model
        if args.provider == "ollama":
            MODEL_NAME = "llama3.2"  # common local Ollama model; override with --model if needed
        elif args.provider == "huggingface":
            MODEL_NAME = _get("SLM_MODEL") or "HuggingFaceH4/zephyr-7b-beta"
        elif args.provider == "local":
            MODEL_NAME = _get("SLM_MODEL") or "google/gemma-3-1b-it"

    logger.info(f"SLM provider: {SLM_PROVIDER}  |  model: {MODEL_NAME}  |  output: {OUTPUT_PATH}")
    if not INPUT_PATH.exists():
        logger.error(f"{INPUT_PATH} not found.")
        return

    df = read_parquet_safe(str(INPUT_PATH))
    results_df = read_parquet_safe(str(OUTPUT_PATH)) if OUTPUT_PATH.exists() else pd.DataFrame()
    labeled_ids = set(results_df["id"].tolist()) if not results_df.empty else set()

    df_remaining = df[~df["id"].isin(labeled_ids)].copy()
    to_process = len(df_remaining)
    logger.info(f"Process strategy: {to_process} / {len(df)} remaining")

    if to_process == 0:
        logger.info("Nothing to do.")
        return

    client = get_client()
    if not verify_model_responds(client, df_remaining.iloc[0] if not df_remaining.empty else None):
        return

    use_optimized_prompt = False  # Set True for providers that use the optimized prompt
    batch = []
    done_in_run = 0
    for idx, row in df_remaining.iterrows():
        prompt = construct_prompt_optimized(row) if use_optimized_prompt else construct_prompt(row)
        raw = call_slm_with_retries(client, prompt)
        parsed = parse_slm_response(raw)
        
        new_row = row.to_dict()
        new_row.update(parsed)
        batch.append(new_row)
        done_in_run += 1
        
        logger.info(f"[{done_in_run}/{to_process}] id={str(row['id'])[:8]}... name={parsed['attr_name_winner']}")

        if len(batch) >= SAVE_EVERY_N:
            results_df = pd.concat([results_df, pd.DataFrame(batch)], ignore_index=True)
            _save_results(results_df, OUTPUT_PATH)
            logger.info(f"Checkpoint: {len(results_df)} rows saved.")
            batch = []

    if batch:
        results_df = pd.concat([results_df, pd.DataFrame(batch)], ignore_index=True)
        _save_results(results_df, OUTPUT_PATH)
    
    logger.info(f"Complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

