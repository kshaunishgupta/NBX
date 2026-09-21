# NBX - Next Best Experience (reference folder)

> **The code has moved.** `nbx_all_in_one_rich.py` was consolidated into the main
> project directory alongside the Streamlit app:
>
> ```
> /Users/apple/Desktop/Co-op/Y1 Spring - OFTI/Co-op/NBX/NBX/
> ```
>
> This folder now holds reference material only: the architecture diagram, the
> original PoC output CSVs, and the source papers in `../External Sources/`.

LLM "Next Best Experience" layer that sits ON TOP of the existing NBA contextual-bandit
recommender - WITHOUT changing the recommender or CVM.
  - Recommender decides WHAT (unchanged).
  - NBX adds WHY (grounded rationale) + HOW (channel-adapted message).
  - Margin & Policy Guard validates every message (read-only).

## Run (from the new location)
```bash
cd "/Users/apple/Desktop/Co-op/Y1 Spring - OFTI/Co-op/NBX/NBX"
pip3 install -r requirements.txt
python3 nbx_all_in_one_rich.py
```
Outputs are written to `./outputs/all_in_one/` — a subfolder, so this PoC's CSVs
stay separate from the Streamlit pipeline's CSVs in the same parent directory.

## Vector retrieval layer (optional)
Instead of stuffing a customer's entire unstructured text into every prompt,
embed their interaction history and retrieve only the most decision-relevant
snippets.

```bash
NBX_VECTOR_RETRIEVAL=1 python3 nbx_all_in_one_rich.py   # enable
python3 notebooks/vector_db_learning.py                 # see how it performs
```

Measured on synthetic data with ground-truth relevance labels:

| Metric | Value |
|---|---|
| precision@3 | 78.9% |
| recall@3 | 80.3% |
| random baseline | 37.4% |
| prompt payload reduction | ~60% fewer characters |

Stack: `sentence-transformers` (all-MiniLM-L6-v2, local, no API key) + `chromadb`
(in-memory). See `pipeline/vector_store.py` for why per-customer ranking is done
exactly rather than through the approximate HNSW index.

## Wire to a real LLM (env vars only, no code change)
```bash
export NBX_LLM_PROVIDER=openai
export NBX_LLM_BASE_URL=https://api.openai.com/v1
export NBX_LLM_API_KEY=sk-...
export NBX_LLM_MODEL=gpt-4o-mini
# local / in-tenant (data residency):
# export NBX_LLM_PROVIDER=local
# export NBX_LLM_BASE_URL=http://localhost:11434/v1
# export NBX_LLM_MODEL=llama3.1
```
If unset, an offline deterministic stand-in runs automatically.

## Data (20,000 rows x 54 fields)
Source systems: BSS/CRM, Billing, Consumption, Device & Network, Engagement,
Care/Unstructured (incl. call transcript + sentiment), and NBA/NBO/CVM outputs.
Includes device type/tier/capability and full network-experience signals
(speed, latency, dropped-call rate, coverage, congestion, complaints) with a
derived Network_Issue_Flag (17.2% of base). When a network issue is detected,
NBX leads with empathy and avoids a hard upsell.

## Proven value (PoC simulation, sample 800 customers)
| Metric | Value |
|---|---|
| Acceptance baseline | 22.7% |
| Acceptance NBX | 29.8% |
| Revenue uplift | +32.2% |
| Extrapolated +QAR/yr @2M | 409,556,796 |

Figures are a defensible simulation on synthetic data, to be confirmed via A/B holdout.

## Safety
- Action never changes (bandit stays system of record for WHAT).
- Strict grounding: numeric claims must match context, else safe fallback.
- PII redaction on transcript before any hosted-LLM call (also applied to
  retrieved snippets).
- Margin & Policy Guard on every message; graceful LLM fallback.

## Files
### Now in `NBX/NBX/` (the main project)
- `nbx_all_in_one_rich.py`         - the full single-file PoC
- `data/interaction_history.py`    - synthetic unstructured interaction histories
- `pipeline/vector_store.py`       - embeddings + Chroma retrieval
- `notebooks/vector_db_learning.py`- retrieval quality comparison + write-up
- `requirements.txt`
- `outputs/all_in_one/`            - generated CSVs

### Still here (reference only)
- `nbx_architecture.png`           - architecture diagram
- `nba_recommendations.csv`, `nbx_experience.csv` - original PoC outputs
- `outputs/`, `outputs_backup_pre_vector/`        - earlier generated CSVs
- `HOW_TO_RUN.txt`
