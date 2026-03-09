"""
SLM per-attribute labeler: for each place pair in phase1_processed.parquet,
use an SLM to choose the better source (base / alt / both / none) for each
attribute (name, phone, web, address, category). No "same place" logic—only
attribute-level superiority. Output schema matches golden_dataset_maker.
Run from project root: python scripts/slm_attribute_labeler.py
"""
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from parquet_io import read_parquet_safe

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "phase1_processed.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "phase3_slm_labeled.parquet"
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


# Provider selection (Hugging Face only):
# - "local": AutoTokenizer + AutoModelForCausalLM on this machine (e.g. google/gemma-3-1b-it)
# - "huggingface": HF_TOKEN + router.huggingface.co (OpenAI-compatible API)
_has_hf = bool(_get("HF_TOKEN") or _get("HUGGINGFACE_API_KEY"))
_provider_default = "huggingface" if _has_hf else "local"
SLM_PROVIDER = _get("SLM_PROVIDER", _provider_default).lower()
if SLM_PROVIDER not in ("local", "huggingface", "ollama"):
    SLM_PROVIDER = "local"

HF_TOKEN = _get("HF_TOKEN") or _get("HUGGINGFACE_API_KEY")
HF_ROUTER_URL = "https://router.huggingface.co/v1"

# Ollama (local OpenAI-compatible server for kimi-k2-instruct-0905)
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = _get("OLLAMA_API_KEY", "ollama")

# Model name: use SLM_MODEL if set; otherwise default by provider
if SLM_PROVIDER == "local":
    _default_model = "google/gemma-3-1b-it"
elif SLM_PROVIDER == "huggingface":
    _default_model = "HuggingFaceH4/zephyr-7b-beta"
elif SLM_PROVIDER == "ollama":
    _default_model = "kimi-k2-instruct-0905"
else:
    _default_model = "kimi-k2-instruct-0905"

MODEL_NAME = _get("SLM_MODEL") or _default_model

# Match golden_dataset_maker schema
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
BOOKKEEPING_COLUMNS = ["golden_label"] + [f"attr_{a}_winner" for a in ATTR_ATTRS]


# Prompt template for same-place reasoning (kimi-k2-instruct-0905 via Ollama)
PROMPT_TEMPLATE = """
Determine whether these two business records describe the same physical location.

RULES:
- Confidence > 0.5 suggests they are likely the same place.
- Focus on street number and street name; these are the strongest signals.
- Same ZIP/postcode = same general area; treat as supporting "same place."
- Suite/unit/floor differences only = still the SAME place.
- Ignore category, brand, phone, and website; they often differ for the same location.

Output: 0 = different places, 1 = same place.

Record A:
Name: {name}
Address: {address_full}
City: {locality}, {region} {postcode}
Country: {country}

Record B:
Name: {base_name}
Address: {base_address_full}
City: {base_locality}, {base_region} {base_postcode}
Country: {base_country}
"""


def _winner_str(w):
    """Normalize winner to 'base', 'alt', 'both', or 'none' for bookkeeping."""
    return w if w in ("base", "alt", "both") else "none"


# --- Display helpers (aligned with golden_dataset_maker so SLM sees same values) ---
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
    """Build a prompt that first frames same-place reasoning, then asks for per-attribute winners."""
    displays = get_display_values(row)
    base_name, alt_name = displays["name"]
    base_addr, alt_addr = displays["address"]

    record_prompt = PROMPT_TEMPLATE.format(
        # Treat the conflated record as Record A, base as Record B
        name=alt_name,
        address_full=alt_addr,
        locality=row.get("locality") or "",
        region=row.get("region") or "",
        postcode=row.get("postcode") or "",
        country=row.get("country") or "",
        base_name=base_name,
        base_address_full=base_addr,
        base_locality=row.get("base_locality") or "",
        base_region=row.get("base_region") or "",
        base_postcode=row.get("base_postcode") or "",
        base_country=row.get("base_country") or "",
    )

    blocks = []
    for attr in ATTR_ATTRS:
        base_val, alt_val = displays[attr]
        blocks.append(f"  {attr}:  base = {base_val!r}   alt = {alt_val!r}")
    attr_block = "\n".join(blocks)

    return f"""{record_prompt}

You are a data steward. For this same pair of records, we have two sources: "base" and "alt" (conflated). For each attribute below, choose the BETTER SOURCE: base, alt, both, or none.
- base = base record's value is better or only valid
- alt = alt record's value is better or only valid
- both = equivalent or both acceptable
- none = neither is useful (both empty/invalid)

Attribute values (base vs alt):
{attr_block}

Respond with ONLY a JSON object (no markdown). Use exactly these keys and values in {{"base","alt","both","none"}} per attribute. Optionally include "golden_label" (one of: base, alt, abstain) and "reason" (short sentence).
Example: {{"name":"both","phone":"base","web":"alt","address":"base","category":"both","golden_label":"base","reason":"..."}}"""


def get_client():
    """Return a client object appropriate for the selected provider."""
    if SLM_PROVIDER == "local":
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print("ERROR: For provider=local install transformers and torch: pip install transformers torch")
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
            print("WARNING: No Hugging Face token. Set HF_TOKEN or HUGGINGFACE_API_KEY in env or api_keys.env.")
            return None
        return OpenAI(api_key=HF_TOKEN, base_url=HF_ROUTER_URL)

    if SLM_PROVIDER == "ollama":
        # Use local Ollama server exposing an OpenAI-compatible API
        return OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

    return None


def _extract_json(text: str):
    """Try to parse JSON from model output (may be wrapped in markdown or extra text)."""
    text = (text or "").strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON found in model output")


def call_slm(client, prompt, model=MODEL_NAME):
    """Call SLM; return parsed JSON or empty dict on failure."""
    if not client:
        return {
            "name": "none", "phone": "none", "web": "none", "address": "none", "category": "none",
            "golden_label": "abstain", "reason": "API key missing (mock)."
        }
    try:
        # Local: AutoTokenizer + AutoModelForCausalLM with chat template and generate
        if SLM_PROVIDER == "local":
            tokenizer = client["tokenizer"]
            model = client["model"]
            full_prompt = (
                "You output only valid JSON with keys: name, phone, web, address, category "
                "(each value: base, alt, both, or none). Optionally golden_label (base/alt/abstain) and reason.\n\n"
                + prompt
            )
            messages = [{"role": "user", "content": full_prompt}]
            # Get prompt string from chat template, then tokenize explicitly to avoid shape issues (e.g. Gemma 3)
            try:
                prompt_str = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                prompt_str = full_prompt
            if isinstance(prompt_str, list):
                prompt_str = prompt_str[0] if prompt_str else full_prompt
            encoded = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            )
            pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            else:
                # When pad_token_id == eos_token_id, tokenizer may omit mask; pass all-ones for single sequence
                import torch
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
            gen_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "max_new_tokens": 512}
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
            outputs = model.generate(**gen_kwargs)
            prompt_len = input_ids.shape[1]
            content = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return _extract_json(content)
        # Hugging Face router (OpenAI-compatible chat.completions)
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You output only valid JSON with keys: name, phone, web, address, category (each value: base, alt, both, or none). Optionally golden_label (base/alt/abstain) and reason."},
                {"role": "user", "content": prompt},
            ],
        }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return _extract_json(content)
    except Exception as e:
        raise RuntimeError(str(e))


def call_slm_with_retries(client, prompt, model=MODEL_NAME):
    """Call SLM with exponential backoff. Returns dict with attr winners or default none."""
    default = {
        "name": "none", "phone": "none", "web": "none", "address": "none", "category": "none",
        "golden_label": "abstain", "reason": "",
    }
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            out = call_slm(client, prompt, model=model)
            return out
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                default["reason"] = f"Error after {MAX_RETRIES} retries: {e}"
                return default
            time.sleep(backoff)
            backoff *= 2
    return default


def parse_slm_response(raw):
    """Map SLM JSON to attr_*_winner and golden_label; normalize to base/alt/both/none."""
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
    """
    Run one test call and check the model returns at least one non-none attribute.
    Returns True if the model is usable, False otherwise.
    """
    if client is None:
        print("Verification failed: no API client (missing key or invalid provider).")
        return False
    # Use a minimal prompt if no row provided (e.g. when to_process == 0 we still verify)
    if sample_row is not None:
        prompt = construct_prompt(sample_row)
    else:
        prompt = """You are a data steward. For each attribute choose the better source: base, alt, both, or none.
  name:  base = 'Cafe A'   alt = 'Cafe A'
  phone:  base = '(555) 123-4567'   alt = '(empty)'
  web:   base = '(empty)'   alt = 'https://example.com'
  address: base = '123 Main St'   alt = '123 Main Street'
  category: base = 'Restaurant'   alt = 'Restaurant'

Respond with ONLY a JSON object. Keys: name, phone, web, address, category (values: base, alt, both, none). Optionally golden_label and reason.
Example: {"name":"both","phone":"base","web":"alt","address":"both","category":"both","golden_label":"base","reason":"..."}"""
    try:
        raw = call_slm(client, prompt)
    except Exception as e:
        err_str = str(e)
        print(f"Verification failed: model call raised: {e}")
        if "model_not_supported" in err_str or "not supported by any provider" in err_str:
            print("Hint: For Hugging Face router, pick a model supported by your enabled providers.")
            print("  Try removing SLM_MODEL to use the default, or set e.g. SLM_MODEL=HuggingFaceH4/zephyr-7b-beta")
        return False
    parsed = parse_slm_response(raw)
    non_none = [a for a in ATTR_ATTRS if parsed.get(f"attr_{a}_winner") != "none"]
    if not non_none:
        print("Verification failed: model returned all attributes as 'none'. Check provider, model, and prompt.")
        print("Raw response:", raw)
        return False
    print(f"Verification OK: model responded with non-none labels for {', '.join(non_none)} (provider={SLM_PROVIDER}, model={MODEL_NAME})")
    return True


def main():
    print(f"SLM provider: {SLM_PROVIDER}  |  model: {MODEL_NAME}")
    print("Loading phase1 data...")
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return
    df = read_parquet_safe(str(INPUT_PATH))
    total = len(df)

    # Resume: skip already-labeled ids
    labeled_ids = set()
    results_df = pd.DataFrame()
    if OUTPUT_PATH.exists():
        results_df = read_parquet_safe(str(OUTPUT_PATH))
        labeled_ids = set(results_df["id"].tolist())
        print(f"Resuming: {len(labeled_ids)} rows already labeled in {OUTPUT_PATH}")
    df_remaining = df[~df["id"].isin(labeled_ids)].copy()
    to_process = len(df_remaining)
    print(f"Rows to process: {to_process} / {total}")

    if to_process == 0:
        print("Nothing to do.")
        return

    client = get_client()
    # Verify model is reachable and returns at least one non-none attribute before processing
    sample_row = df_remaining.iloc[0] if len(df_remaining) > 0 else None
    if not verify_model_responds(client, sample_row):
        print("Exiting. Fix API key, provider, or model and run again.")
        return
    batch = []
    done_in_run = 0

    for idx, row in df_remaining.iterrows():
        prompt = construct_prompt(row)
        try:
            raw = call_slm_with_retries(client, prompt)
        except Exception as e:
            raw = {
                "name": "none", "phone": "none", "web": "none", "address": "none", "category": "none",
                "golden_label": "abstain", "reason": str(e),
            }
        parsed = parse_slm_response(raw)
        new_row = row.to_dict()
        for k, v in parsed.items():
            new_row[k] = v
        batch.append(new_row)
        done_in_run += 1
        row_id = row["id"]
        print(f"  {done_in_run}/{to_process} id={str(row_id)[:12]}... name={parsed['attr_name_winner']}")

        if len(batch) >= SAVE_EVERY_N:
            append_df = pd.DataFrame(batch)
            results_df = pd.concat([results_df, append_df], ignore_index=True)
            _save_results(results_df, OUTPUT_PATH)
            print(f"  [Saved {len(results_df)} rows]")
            batch = []

    if batch:
        append_df = pd.DataFrame(batch)
        results_df = pd.concat([results_df, append_df], ignore_index=True)
    _save_results(results_df, OUTPUT_PATH)
    print(f"Done. Saved {len(results_df)} rows to {OUTPUT_PATH}")
    _print_summary(results_df)


def _save_results(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with original columns plus bookkeeping (golden_label, attr_*_winner, SLM_reason)."""
    other = [c for c in df.columns if c not in BOOKKEEPING_COLUMNS and c != "SLM_reason"]
    ordered = other + [c for c in BOOKKEEPING_COLUMNS if c in df.columns]
    if "SLM_reason" in df.columns:
        ordered.append("SLM_reason")
    df[ordered].to_parquet(path, index=False)


def _print_summary(df: pd.DataFrame) -> None:
    """Print counts of base/alt/both/none per attribute and golden_label distribution."""
    print("\n--- Summary ---")
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner"
        if col in df.columns:
            print(df[col].value_counts().to_string())
            print()
    if "golden_label" in df.columns:
        print("golden_label:")
        print(df["golden_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
