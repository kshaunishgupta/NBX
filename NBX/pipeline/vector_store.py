#!/usr/bin/env python3
"""
================================================================================
VECTOR STORE  --  embeddings + Chroma retrieval  (Step 3 of the learning module)
================================================================================
WHAT THIS REPLACES
------------------
Before: build_context() took the customer's ENTIRE unstructured text and stuffed
        all of it into the LLM prompt, every single time.
After:  we embed every historical interaction once, then at prompt-build time we
        ask "which 3 of this customer's interactions actually matter for a
        recommendation decision?" and send only those.

THE THREE IDEAS WORTH UNDERSTANDING
-----------------------------------
1. EMBEDDING
   `all-MiniLM-L6-v2` turns a sentence into a 384-dimensional vector. Sentences
   with similar *meaning* land near each other, even with no shared words.
   "Videos keep freezing every evening" and "network issues" share zero
   keywords, but their vectors are close. That is the whole trick.

2. THE VECTOR INDEX (Chroma)
   Chroma stores {vector, document, metadata} and answers "nearest neighbours to
   this query vector". We attach customer_id as metadata and use a `where`
   filter so customer A can never retrieve customer B's history -- in telco that
   is a hard privacy boundary, not an optimisation.

3. DISTANCE vs SIMILARITY  (the easiest thing to get wrong)
   Chroma's DEFAULT distance is squared L2, NOT cosine. On L2 the numbers are
   unbounded and "0.31" means nothing intuitive. We explicitly create the
   collection with cosine space, which gives distance in [0, 2], so:
                       similarity = 1 - distance
   1.0 = identical meaning, 0.0 = unrelated, <0 = opposite.
   (Verified empirically against chromadb 1.5.9: an identical vector returns
   distance exactly 0.0.)

All data is synthetic. No real customer data.
================================================================================
"""

from __future__ import annotations

import time

# ── The query that represents "what NBX cares about" ──────────────────────────
# This is the semantic anchor. We are not keyword-matching it; we embed it and
# find interactions whose MEANING sits near it.
NBX_QUERY = "customer complaints, churn signals, network issues, upgrade interest"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small (~90MB), fast, CPU-friendly, free
COLLECTION_NAME = "customer_interactions"

_model = None          # cached SentenceTransformer (loading is slow; do it once)
_default_store = None  # module-level store so retrieve_relevant_context() works


def get_embedding_model():
    """Load the local embedding model once and reuse it (first call downloads it)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip3 install sentence-transformers chromadb"
            ) from e
        t0 = time.perf_counter()
        print(f"   [embed] loading model '{EMBED_MODEL_NAME}' ...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        dim = _model.get_sentence_embedding_dimension()
        print(f"   [embed] ready in {time.perf_counter() - t0:.2f}s  ({dim} dimensions per sentence)")
    return _model


class VectorStore:
    """Thin, readable wrapper around a Chroma collection of customer interactions."""

    def __init__(self, collection, model, n_indexed: int):
        self.collection = collection
        self.model = model
        self.n_indexed = n_indexed

    # -- retrieval ------------------------------------------------------------
    def retrieve(self, customer_id: str, query: str = NBX_QUERY, top_k: int = 3,
                 exact: bool = True) -> list[dict]:
        """
        Return this customer's top_k most semantically relevant interactions.

        The customer_id filter is the privacy boundary: a customer can only ever
        retrieve their own interactions.

        WHY `exact=True` IS THE DEFAULT  (a real bug this module hit)
        ------------------------------------------------------------
        Chroma's query() uses HNSW, an APPROXIMATE nearest-neighbour index.
        Our corpus is built from a small pool of template sentences, so ~6,700
        interactions collapse onto only ~31 distinct embedding vectors --
        thousands of *identical* points. HNSW builds its graph from vector
        distances, and huge clusters of identical vectors degenerate that graph:
        filtered queries then returned EMPTY for 131 of 150 customers, and gave
        different answers on repeated identical calls.

        Measured, not guessed:
            real MSISDNs, no duplicate transcripts -> 0/150 customers empty
            + 666 records sharing 10 strings       -> 131/150 customers empty

        Note it is INTERMITTENT -- a later run of the same ANN query returned
        0/150 empty. Silent, load-dependent wrongness is worse than a crash:
        you would ship it, and simply never see those customers' history.

        The fix is not to fight the index. Each customer has at most ~11
        interactions, so approximate search buys nothing: we fetch that
        customer's rows by metadata and rank them EXACTLY with cosine
        similarity. Exact, deterministic, and still sub-millisecond.

        At genuine scale (millions of rows per query scope) you WOULD need the
        ANN index -- and you would also have real, non-duplicated text, which is
        the condition HNSW actually assumes. Pass exact=False to use it.
        """
        q_vec = self.model.encode([query], normalize_embeddings=True)

        if not exact:
            # ---- approximate path: Chroma's HNSW index -----------------------
            res = self.collection.query(
                query_embeddings=q_vec.tolist(),
                n_results=top_k,
                where={"customer_id": {"$eq": str(customer_id)}},
                include=["documents", "metadatas", "distances"],
            )
            if not res["ids"] or not res["ids"][0]:
                return []
            return [{
                "interaction_id": iid,
                "text": doc,
                "date": meta.get("date"),
                "channel": meta.get("channel"),
                "customer_id": meta.get("customer_id"),
                "distance": round(float(dist), 4),
                "similarity": round(1.0 - float(dist), 4),
            } for iid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

        # ---- exact path: pull this customer's rows, score them directly ------
        import numpy as np

        got = self.collection.get(
            where={"customer_id": {"$eq": str(customer_id)}},
            include=["embeddings", "documents", "metadatas"],
        )
        if not got["ids"]:
            return []

        mat = np.asarray(got["embeddings"], dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        # vectors were normalised at insert time; renormalise defensively so the
        # dot product is a true cosine similarity
        mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
        qv = q_vec[0] / max(float(np.linalg.norm(q_vec[0])), 1e-12)
        sims = mat @ qv

        order = np.argsort(-sims)[:top_k]
        out = []
        for idx in order:
            meta = got["metadatas"][idx]
            sim = float(sims[idx])
            out.append({
                "interaction_id": got["ids"][idx],
                "text": got["documents"][idx],
                "date": meta.get("date"),
                "channel": meta.get("channel"),
                "customer_id": meta.get("customer_id"),
                "distance": round(1.0 - sim, 4),
                "similarity": round(sim, 4),
            })
        return out

    def explain(self, customer_id: str, all_interactions: list[dict],
                query: str = NBX_QUERY, top_k: int = 3) -> dict:
        """
        Retrieval WITH its counterfactual: what was retrieved vs what was left
        behind. Seeing the rejected items is what makes retrieval quality legible.
        """
        retrieved = self.retrieve(customer_id, query=query, top_k=top_k)
        kept_ids = {r["interaction_id"] for r in retrieved}
        skipped = [i for i in all_interactions
                   if str(i["customer_id"]) == str(customer_id)
                   and i["interaction_id"] not in kept_ids]
        return {"retrieved": retrieved, "not_retrieved": skipped}


def build_vector_store(all_interactions: list[dict], verbose: bool = True) -> VectorStore:
    """
    Embed every interaction and load it into an in-memory Chroma collection.

    Args:
        all_interactions: dicts with interaction_id, customer_id, date, channel, text
    Returns:
        VectorStore (also registered as the module-level default store)
    """
    global _default_store

    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb is not installed.\n  pip3 install chromadb sentence-transformers") from e

    if not all_interactions:
        raise ValueError("build_vector_store() got an empty interaction list.")

    model = get_embedding_model()

    # ---- 1. embed everything in one batched call (far faster than per-row) ---
    texts = [i["text"] for i in all_interactions]
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    t_embed = time.perf_counter() - t0

    # ---- 2. create the collection in COSINE space (not the L2 default) -------
    client = chromadb.Client()                      # in-memory / ephemeral
    try:
        client.delete_collection(COLLECTION_NAME)   # idempotent across re-runs
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    # ---- 3. add vectors + metadata, in batches ------------------------------
    # Chroma rejects a single add() larger than its max batch size (5461 on
    # chromadb 1.5.9). That limit is invisible on toy data and only bites once
    # you scale up -- 800 customers x ~8 interactions = 6059 rows blows past it.
    # Ask the client for its actual limit rather than hardcoding a version-
    # specific number.
    ids_all = [i["interaction_id"] for i in all_interactions]
    embs_all = embeddings.tolist()
    metas_all = [{
        "customer_id": str(i["customer_id"]),
        "date": i["date"],
        "channel": i["channel"],
    } for i in all_interactions]

    try:
        max_batch = int(client.get_max_batch_size())
    except Exception:
        max_batch = 5000  # conservative fallback if the API is unavailable
    batch = max(1, min(max_batch - 1, 5000))

    t1 = time.perf_counter()
    n_batches = 0
    for start in range(0, len(ids_all), batch):
        stop = start + batch
        collection.add(
            ids=ids_all[start:stop],
            embeddings=embs_all[start:stop],
            documents=texts[start:stop],
            metadatas=metas_all[start:stop],
        )
        n_batches += 1
    t_add = time.perf_counter() - t1

    if verbose:
        n_cust = len({i["customer_id"] for i in all_interactions})
        print(f"   [store] embedded {len(texts)} interactions from {n_cust} customers "
              f"in {t_embed:.2f}s  ({len(texts)/max(t_embed,1e-9):.0f}/sec)")
        print(f"   [store] indexed into Chroma in {t_add:.2f}s  "
              f"({n_batches} batch{'es' if n_batches != 1 else ''} of <= {batch})  "
              f"|  collection count = {collection.count()}")

    _default_store = VectorStore(collection, model, len(texts))
    return _default_store


def retrieve_relevant_context(customer_id: str, query: str = NBX_QUERY, top_k: int = 3) -> list[dict]:
    """
    Module-level convenience matching the requested signature.
    Requires build_vector_store() to have been called first.
    """
    if _default_store is None:
        raise RuntimeError("No vector store built yet -- call build_vector_store(...) first.")
    return _default_store.retrieve(customer_id, query=query, top_k=top_k)


def get_default_store() -> "VectorStore | None":
    """Return the active store (None if not built) -- used for graceful fallback."""
    return _default_store


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.interaction_history import generate_all_histories

    print("Building a small demo vector store ...")
    ids = [f"9745000{i:04d}" for i in range(20)]
    corpus = generate_all_histories(ids)
    store = build_vector_store(corpus)

    cid = ids[0]
    print(f"\nTop 3 for {cid}  (query: '{NBX_QUERY}')")
    for r in store.retrieve(cid):
        print(f"   sim={r['similarity']:+.3f}  {r['date']}  {r['channel']:<9} {r['text']}")
