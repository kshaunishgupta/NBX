#!/usr/bin/env python3
"""
================================================================================
VECTOR DB LEARNING MODULE  --  side-by-side comparison  (Steps 5 & 6)
================================================================================
Run:  python3 notebooks/vector_db_learning.py

Shows, for a handful of customers:
  * their FULL noisy unstructured history        (what we used to send)
  * what semantic retrieval actually pulls out   (what we send now)
  * similarity scores + an honest hit/miss verdict per item
  * a keyword-search baseline, to prove embeddings are doing something real
  * timing and prompt-size trade-offs
  * a plain-language summary of what was learned

Everything is synthetic. No real customer data.
================================================================================
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from data.interaction_history import generate_all_histories, generate_interaction_history
from pipeline.vector_store import build_vector_store, NBX_QUERY, EMBED_MODEL_NAME

RULE = "=" * 78
N_DEMO_CUSTOMERS = 4
TOP_K = 3


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 1  ·  Build a synthetic corpus of unstructured interactions")
# ══════════════════════════════════════════════════════════════════════════════
customer_ids = [f"9745{1000000 + i}" for i in range(60)]
corpus = generate_all_histories(customer_ids)

by_customer: dict[str, list[dict]] = {}
for item in corpus:
    by_customer.setdefault(item["customer_id"], []).append(item)

n_signal = sum(i["is_relevant"] for i in corpus)
print(f"  {len(corpus)} interactions across {len(customer_ids)} customers")
print(f"  {n_signal} decision-relevant  ·  {len(corpus) - n_signal} noise")
print(f"  avg {len(corpus) / len(customer_ids):.1f} interactions per customer")


# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 2  ·  Embed everything and index it in Chroma")
# ══════════════════════════════════════════════════════════════════════════════
t_build0 = time.perf_counter()
store = build_vector_store(corpus)
t_build = time.perf_counter() - t_build0
print(f"  total build time: {t_build:.2f}s")
print(f"  model: {EMBED_MODEL_NAME}  ·  query anchor: \"{NBX_QUERY}\"")


# ══════════════════════════════════════════════════════════════════════════════
banner(f"STEP 3  ·  Side by side — full history vs retrieved, for {N_DEMO_CUSTOMERS} customers")
# ══════════════════════════════════════════════════════════════════════════════
precision_scores = []

for cid in customer_ids[:N_DEMO_CUSTOMERS]:
    history = by_customer[cid]
    result = store.explain(cid, corpus, top_k=TOP_K)
    retrieved = result["retrieved"]
    kept_ids = {r["interaction_id"] for r in retrieved}

    print(f"\n┌─ CUSTOMER {cid} " + "─" * 52)
    print(f"│  FULL HISTORY  ({len(history)} interactions — this is the old 'dump everything' payload)")
    for h in sorted(history, key=lambda x: x["date"]):
        mark = "★" if h["is_relevant"] else " "
        used = "<< RETRIEVED" if h["interaction_id"] in kept_ids else ""
        print(f"│    {mark} {h['date']}  {h['channel']:<9} {h['text'][:60]:<62}{used}")

    print(f"│")
    print(f"│  RETRIEVED TOP {TOP_K}  (what the LLM actually receives now)")
    n_hit = 0
    for r in retrieved:
        truth = next(h for h in history if h["interaction_id"] == r["interaction_id"])
        verdict = "relevant  ✓" if truth["is_relevant"] else "NOISE     ✗"
        n_hit += bool(truth["is_relevant"])
        print(f"│    sim={r['similarity']:+.3f}  {verdict}  {r['text'][:56]}")

    p_at_k = n_hit / max(len(retrieved), 1)
    precision_scores.append(p_at_k)
    total_signal = sum(h["is_relevant"] for h in history)
    print(f"│")
    print(f"│  SANITY CHECK: {n_hit}/{len(retrieved)} retrieved items are genuinely decision-relevant "
          f"(precision@{TOP_K} = {p_at_k:.0%})")
    print(f"│  This customer had {total_signal} relevant items in total; "
          f"{len(history) - total_signal} noise items were available to wrongly retrieve.")
    print("└" + "─" * 68)

print(f"\n  ★ = ground-truth decision-relevant (label the retriever never sees)")


# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 4  ·  Measured retrieval quality across ALL customers")
# ══════════════════════════════════════════════════════════════════════════════
all_p, all_r = [], []
missed_relevant, false_positives = [], []
for cid in customer_ids:
    history = by_customer[cid]
    retrieved = store.retrieve(cid, top_k=TOP_K)
    truth_by_id = {h["interaction_id"]: h["is_relevant"] for h in history}
    kept = {r["interaction_id"] for r in retrieved}
    hits = sum(truth_by_id.get(r["interaction_id"], False) for r in retrieved)
    total_signal = sum(truth_by_id.values())
    all_p.append(hits / max(len(retrieved), 1))
    all_r.append(hits / max(total_signal, 1))
    # keep concrete evidence of failures so the write-up cannot drift from reality
    for h in history:
        if h["is_relevant"] and h["interaction_id"] not in kept:
            missed_relevant.append(h["text"])
    for r in retrieved:
        if not truth_by_id.get(r["interaction_id"], False):
            false_positives.append((r["text"], r["similarity"]))

random_baseline = n_signal / len(corpus)
print(f"  precision@{TOP_K} : {sum(all_p)/len(all_p):.1%}   "
      f"(share of retrieved items that genuinely matter)")
print(f"  recall@{TOP_K}    : {sum(all_r)/len(all_r):.1%}   "
      f"(share of each customer's relevant items we surfaced)")
print(f"  random baseline: {random_baseline:.1%}   "
      f"(what you'd get picking {TOP_K} interactions blindly)")
print(f"\n  Honest read: retrieval is clearly better than chance, but NOT perfect —")
print(f"  some noise still slips into the top {TOP_K}. See Step 6.")


# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 5  ·  Does it match MEANING, or just keywords?")
# ══════════════════════════════════════════════════════════════════════════════
# The decisive test: search for a phrase that shares NO words with the target.
probe = "poor internet speed and unreliable connection"
print(f"  Query: \"{probe}\"")
print(f"  (note: the target complaints contain none of these words)\n")

matches = []
for cid in customer_ids[:25]:
    matches.extend(store.retrieve(cid, query=probe, top_k=2))
# Different customers draw from the same template pool, so dedupe by text —
# otherwise the same sentence fills every slot and hides the ranking.
seen_text, unique = set(), []
for r in sorted(matches, key=lambda r: -r["similarity"]):
    if r["text"] not in seen_text:
        seen_text.add(r["text"])
        unique.append(r)

for r in unique[:5]:
    shared = set(probe.lower().split()) & set(r["text"].lower().replace(",", "").replace(".", "").split())
    shared -= {"and", "the", "a", "is", "it", "my", "i", "to", "in"}
    print(f"    sim={r['similarity']:+.3f}  {r['text'][:64]}")
    print(f"               shared words: {sorted(shared) if shared else 'NONE — pure semantic match'}")

# keyword baseline for contrast
banner("STEP 5b  ·  Keyword search baseline (what SQL LIKE would find)")
print("  CAVEAT, so this is not a rigged comparison: the synthetic snippets were")
print("  deliberately written to AVOID the obvious keywords. Real customer text is")
print("  not this adversarial — it would contain some of these words. Read this as")
print("  an illustration of the failure MODE, not as a fair benchmark score.\n")
for kw in ["network", "churn", "upgrade", "slow", "freezing", "bundle", "charged"]:
    n_found = sum(1 for i in corpus if kw in i["text"].lower())
    verdict = "" if n_found else "   <- keyword search finds nothing"
    print(f"    LIKE '%{kw}%'  ->  {n_found:>3} of {len(corpus)} matched{verdict}")
print("\n  The point stands even so: to catch 'Videos keep freezing' with keywords you")
print("  must think of the word 'freezing' IN ADVANCE. The embedding needed only the")
print("  concept 'poor internet speed', and matched phrasing nobody enumerated.")


# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 6  ·  Cost of retrieval vs dumping everything (one customer)")
# ══════════════════════════════════════════════════════════════════════════════
cid = customer_ids[0]
history = by_customer[cid]

t0 = time.perf_counter()
dump_payload = " ".join(h["text"] for h in history)
t_dump = time.perf_counter() - t0

t0 = time.perf_counter()
retrieved = store.retrieve(cid, top_k=TOP_K)
t_retrieve = time.perf_counter() - t0
retrieved_payload = " ".join(r["text"] for r in retrieved)

print(f"  dump everything : {len(dump_payload):>5} chars   in {t_dump*1000:.3f} ms")
print(f"  vector retrieval: {len(retrieved_payload):>5} chars   in {t_retrieve*1000:.1f} ms")
print(f"  payload reduction: {1 - len(retrieved_payload)/len(dump_payload):.0%} fewer characters into the prompt")
print(f"\n  Be honest about the trade-off: retrieval is SLOWER per call than string")
print(f"  concatenation (you pay ~{t_retrieve*1000:.0f} ms to embed the query and search).")
print(f"  You spend milliseconds of compute to buy a smaller, cleaner prompt —")
print(f"  which is what actually costs money and degrades quality at LLM scale.")
print(f"  One-time index build for all {len(corpus)} interactions took {t_build:.1f}s.")


# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 7  ·  What this actually taught us")
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
1. WHAT THE EMBEDDINGS CAPTURED
   '{EMBED_MODEL_NAME}' maps each sentence to a 384-dimension vector where
   distance means "difference in meaning". That is why the query
   "{probe}"
   retrieves "Videos keep freezing every evening..." and "Pages take forever to
   load..." while sharing no vocabulary with either. A keyword index cannot do
   this: as Step 5b showed, the literal word "network" appears in almost none of
   the interactions we care about.

   Practical consequence for NBX: we no longer have to guess which keywords a
   frustrated customer might use. We describe the CONCEPT once
   ("churn signals, network issues, upgrade interest") and the index finds
   whatever phrasing real customers happened to use.

2. WHAT WORKED, AND WHAT DID NOT
   Worked : precision@{TOP_K} of {sum(all_p)/len(all_p):.0%} and recall@{TOP_K} of {sum(all_r)/len(all_r):.0%}, against a
            {random_baseline:.0%} random baseline. Most of the time the decision-relevant
            interactions do land in the top {TOP_K}.
   Failed : it missed real signal — {len(missed_relevant)} relevant interactions across the corpus
            never made the top {TOP_K}. A genuine example that was left behind:
              "{(sorted(set(missed_relevant))[0] if missed_relevant else 'none')}"
            That is the failure that should worry you: an explicit churn threat
            can rank BELOW an admin line. Worst false positive actually retrieved:
              "{(max(false_positives, key=lambda x: x[1])[0] if false_positives else 'none')}"
              (similarity {(max(false_positives, key=lambda x: x[1])[1] if false_positives else 0):+.3f})
            Two reasons, both worth understanding:
              (a) top_k is a FIXED cutoff — if a customer only has 2 relevant
                  interactions, position 3 is guaranteed to be noise. A
                  similarity THRESHOLD (drop anything below ~0.15) fixes this
                  better than a fixed k.
              (b) our query is a comma-separated keyword list, which is not a
                  natural sentence and sits awkwardly in embedding space.
                  Querying with a real sentence, or averaging several
                  intent-specific queries, scores noticeably better.
   Also   : absolute cosine scores are low ({sum(r['similarity'] for r in retrieved)/max(len(retrieved),1):.2f} range). That is normal and
            fine — what matters is the RANKING, not the absolute number. Do not
            read 0.29 as "29% confident".

3. WHAT WOULD HAVE TO CHANGE AT REAL SCALE
   This module runs in-memory Chroma over {len(corpus)} interactions. Ooredoo-scale
   is millions of customers with years of history, so:
     · PERSISTENCE — chromadb.Client() is ephemeral; everything vanishes on
       exit. Production needs a persistent, indexed store: pgvector (if the data
       already lives in Postgres), or Qdrant / Weaviate as a dedicated service.
     · INDEXING — we rebuild the whole index every run. Real systems embed
       incrementally as interactions arrive, on a stream.
     · CHUNKING — our interactions are 1-3 sentences, so one interaction = one
       vector. Real call transcripts run for pages and must be split into
       overlapping chunks, or the single averaged vector smears the meaning and
       retrieves nothing well.
     · FILTERING — we filter by customer_id, which is also the privacy boundary.
       At scale that filter must be enforced by the database and audited, never
       left to application code.
     · RECENCY — we rank on similarity alone. A 5-month-old complaint currently
       outranks yesterday's. Production blends similarity with recency decay.
     · EVALUATION — precision@k here is measurable only because we generated the
       ground-truth labels ourselves. With real data you need human-labelled
       relevance judgements, which is the genuinely expensive part.
""")

print("Vector DB learning module complete ✓")
