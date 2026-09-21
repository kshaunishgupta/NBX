#!/usr/bin/env python3
"""
================================================================================
SYNTHETIC INTERACTION HISTORY  (Step 2 of the vector-DB learning module)
================================================================================
WHY THIS FILE EXISTS
--------------------
The original NBX prototype gives each customer exactly ONE unstructured field
(`Call_Transcript`). With one document per customer there is nothing to retrieve
 -- "top 3 of 1" is a trivial problem, and a vector database would teach you
nothing.

So here we simulate what a real telco actually has: a *history* of many
unstructured touchpoints per customer, spread over months, most of which are
irrelevant noise.

THE MOST IMPORTANT DESIGN DECISION
----------------------------------
Each customer's history deliberately mixes:

  * SIGNAL -- 2-3 interactions that genuinely matter to a churn / upsell
    decision ("thinking of moving to another operator", "buffering every night")
  * NOISE  -- the rest: store hours, payment confirmations, app questions

If every interaction were relevant, retrieval would look perfect no matter how
bad the embeddings were. The noise is what makes retrieval quality *visible*.

A second, subtler decision: the SIGNAL snippets are worded so they share almost
NO keywords with the query we will search with ("network issues, churn signals,
upgrade interest"). A keyword/SQL LIKE search would miss them entirely. Only a
semantic embedding match can find them -- which is exactly the lesson.

All data is synthetic. No real customer data, no real PII.
================================================================================
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

# ── Channels the unstructured data can arrive on ──────────────────────────────
CHANNELS = ("call", "chatbot", "social", "complaint")

# ── SIGNAL: interactions that SHOULD influence a recommendation ───────────────
# Note how these avoid the literal words "network", "churn", "upgrade".
# That is intentional -- it forces semantic (not keyword) matching.
RELEVANT_TEMPLATES = [
    # -- experience / quality pain (semantically = network issue) --------------
    ("Videos keep freezing every evening around nine, it is basically unwatchable.",
     "quality_pain", ("call", "complaint", "social")),
    ("Calls keep cutting out whenever I am at home, I have to step outside to finish them.",
     "quality_pain", ("call", "complaint")),
    ("Pages take forever to load even when the signal bars look full.",
     "quality_pain", ("chatbot", "social")),

    # -- competitive / churn intent -------------------------------------------
    ("A friend of mine pays less and gets twice the allowance elsewhere, I am considering moving.",
     "churn_intent", ("call", "social", "complaint")),
    ("If this is not sorted out soon I will take my number somewhere else.",
     "churn_intent", ("complaint", "call")),
    ("What happens to my number if I decide to leave at the end of my contract?",
     "churn_intent", ("chatbot", "call")),

    # -- device / plan upgrade appetite ---------------------------------------
    ("Do you have any trade-in offers on the latest handsets?",
     "upgrade_interest", ("chatbot", "call")),
    ("My handset is getting old and the battery dies by midday, what are my options?",
     "upgrade_interest", ("call", "chatbot")),
    ("Is there a bigger bundle if I keep running out of allowance before month end?",
     "upgrade_interest", ("chatbot", "call")),

    # -- bill shock (drives both churn risk and retention offers) -------------
    ("My charges jumped a lot last month and I cannot see what caused it.",
     "bill_shock", ("call", "complaint")),
    ("I was charged a fortune while I was travelling and nobody warned me.",
     "bill_shock", ("complaint", "call")),
]

# ── NOISE: real interactions that should NOT drive a recommendation ───────────
NOISE_TEMPLATES = [
    ("Just checking what time your branch opens at the weekend.",
     "admin", ("chatbot", "call", "social")),
    ("Payment received, thanks for confirming.",
     "admin", ("chatbot", "call")),
    ("How do I change the language setting inside the app?",
     "how_to", ("chatbot", "social")),
    ("Can you resend last month's invoice as a PDF please.",
     "admin", ("chatbot", "call")),
    ("Thanks for the quick help earlier today, much appreciated.",
     "pleasantry", ("social", "call")),
    ("The new app design looks nice, well done.",
     "pleasantry", ("social",)),
    ("I updated my email address on the account.",
     "admin", ("chatbot", "call")),
    ("Do you sponsor any local football events this season?",
     "smalltalk", ("social",)),
    ("Wanted to confirm my direct debit date for next month.",
     "admin", ("chatbot", "call")),
    ("Is the store open on Friday morning?",
     "admin", ("chatbot", "social")),
]


def _stable_rng(customer_id: str, salt: str = "") -> "random.Random":
    """
    Deterministic per-customer randomness.

    WHY NOT Python's built-in hash()? Because hash() on strings is salted with a
    per-process random seed (PYTHONHASHSEED), so it returns DIFFERENT values on
    every run. Any "reproducible" sampling built on hash(str) is silently
    nondeterministic. hashlib is stable across processes and machines.
    """
    import random
    digest = hashlib.md5(f"{customer_id}|{salt}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def generate_interaction_history(
    customer_id: str,
    n_interactions: int = 8,
    reference_date: datetime | None = None,
    n_relevant: int | None = None,
) -> list[dict]:
    """
    Build one customer's synthetic unstructured history.

    Args:
        customer_id:    the MSISDN (used as a stable seed, so the same customer
                        always gets the same history across runs)
        n_interactions: how many interactions to generate (the corpus helper
                        varies this 5-10 per customer)
        reference_date: "today" -- interactions are spread over the prior 180 days
        n_relevant:     how many should be decision-relevant SIGNAL. Defaults to
                        3 (or 2 for very short histories), so there is always
                        something worth retrieving AND plenty of noise around it.

    Returns:
        list of dicts: interaction_id, customer_id, date, channel, text,
                       plus 'topic' and 'is_relevant' -- ground-truth labels we
                       keep ONLY so the comparison script can score retrieval.
                       The retriever never sees these.
    """
    rng = _stable_rng(customer_id, "history")
    ref = reference_date or datetime.now()

    n_interactions = max(3, int(n_interactions))
    if n_relevant is None:
        n_relevant = 3 if n_interactions >= 6 else 2
    n_relevant = min(n_relevant, n_interactions)
    n_noise = n_interactions - n_relevant

    chosen: list[tuple[str, str, tuple, bool]] = []
    for text, topic, chans in rng.sample(RELEVANT_TEMPLATES, k=min(n_relevant, len(RELEVANT_TEMPLATES))):
        chosen.append((text, topic, chans, True))
    for text, topic, chans in rng.sample(NOISE_TEMPLATES, k=min(n_noise, len(NOISE_TEMPLATES))):
        chosen.append((text, topic, chans, False))

    # Spread across the last 180 days, then sort oldest -> newest.
    day_offsets = sorted(rng.sample(range(1, 181), k=len(chosen)), reverse=True)

    history = []
    for i, ((text, topic, chans, is_rel), days_ago) in enumerate(zip(chosen, day_offsets)):
        history.append({
            "interaction_id": f"{customer_id}-{i:02d}",
            "customer_id": str(customer_id),
            "date": (ref - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "channel": rng.choice(chans),
            "text": text,
            # ground-truth labels: for evaluation only, never fed to the retriever
            "topic": topic,
            "is_relevant": is_rel,
        })
    return history


def generate_all_histories(
    customer_ids,
    min_interactions: int = 5,
    max_interactions: int = 10,
    reference_date: datetime | None = None,
) -> list[dict]:
    """Flatten histories for many customers into one corpus (what we embed)."""
    corpus: list[dict] = []
    for cid in customer_ids:
        rng = _stable_rng(str(cid), "count")
        n = rng.randint(min_interactions, max_interactions)
        corpus.extend(
            generate_interaction_history(str(cid), n_interactions=n, reference_date=reference_date)
        )
    return corpus


if __name__ == "__main__":
    demo_id = "97451234567"
    hist = generate_interaction_history(demo_id, n_interactions=8)
    print(f"Synthetic history for {demo_id}  ({len(hist)} interactions)\n")
    for h in hist:
        tag = "SIGNAL" if h["is_relevant"] else "noise "
        print(f"  [{tag}] {h['date']}  {h['channel']:<9} {h['text']}")
    n_sig = sum(h["is_relevant"] for h in hist)
    print(f"\n  {n_sig} decision-relevant, {len(hist) - n_sig} noise")
