# External Validation

Fetch real-world place data from the internet and compare with base vs alternate (conflated) to produce per-attribute **truth winners** and **suggested values**. Used to verify the golden dataset and to suggest corrections.

## Data

- **Input**: `data/golden_dataset_200.parquet` (hand-labeled golden set; same schema as project_a with base/conflated pairs and optional `attr_*_winner`, `3class_testlabels`).
- **Output**: Same rows plus columns:
  - `truth_phone_winner`, `truth_phone_value`
  - `truth_web_winner`, `truth_web_value`
  - `truth_category_winner`, `truth_category_value`
  - `truth_address_winner`, `truth_address_value`
  - `truth_source` (`google_places`, `scrape_and_search`, or `rule_based`)

Winners are one of: `base`, `alt`, `both`, `real`. `real` means the internet source suggests a value that differs from both base and alternate (suggested update).

## Pipeline A: Google Maps / Places API

- **Script**: `fetch_truth_google.py`
- **Run** (from repo root):  
`python external_validation/fetch_truth_google.py [--limit N] [--dry-run]`
- **Requires**: `GOOGLE_PLACES_API_KEY` in environment or in repo root `api_keys.env` (same style as `scripts/slm_attribute_labeler.py`).
- **Behavior**: Text Search (New) by place name + address → Place Details (New) for phone, website, address, types. Results are cached under `.cache/google_places` and requests are throttled (1/sec) for free tier.
- **Output**: `data/ground_truth_google_golden.parquet`

## Pipeline B: BeautifulSoup + non-Google search

- **Script**: `fetch_truth_scrape.py`
- **Run** (from repo root):  
`python external_validation/fetch_truth_scrape.py [--limit N]`
- **Behavior**: For each row, (1) scrape base or conflated website URL if present (BeautifulSoup), (2) run DuckDuckGo text search with "name + address", (3) merge scraped and search result (scraped wins when present), (4) compare with base/alt and set truth columns.
- **Output**: `data/ground_truth_scrape_golden.parquet`
- **Disclaimer**: Scraping search results may conflict with ToS. Use for research/prototyping only.

## Rule-based (no external data)

- **Script**: `rule_based_logic.py`
- **Run** (from repo root):  
`python external_validation/rule_based_logic.py [--limit N] [--prefer-alt]`
- **Behavior**: Chooses between base and alternate using the same comparison logic as Pipeline A but with no API calls; `--prefer-alt` uses alt values as “real” so alt wins when base and alt disagree.
- **Output**: `data/rule_based.parquet` (same schema; `truth_source` = `rule_based`).

## Verification

- **Script**: `verify_truth.py`
- **Run**:  
`python external_validation/verify_truth.py [--input data/ground_truth_google_golden.parquet]`
- Compares `truth_*_winner` to golden `attr_*_winner` and (if present) record-level `3class_testlabels`; prints agreement counts and sample disagreements.

## Dependencies

- `requests`, `beautifulsoup4`, `diskcache` (already in project).
- `duckduckgo-search` (added for Pipeline B).
- For Pipeline A, no extra install beyond an API key.

## Env / config

- **Google Places**: `GOOGLE_PLACES_API_KEY` in env or in `api_keys.env` at repo root.
- Free tier limits apply; cache and throttle are used to reduce calls.

