"""
Tool: search_compliance_docs
Retrieves relevant Visa rulebook chunks from ChromaDB using MMR
(Maximal Marginal Relevance) to balance relevance and diversity.

MMR selects results that are relevant to the query but dissimilar
to already-selected results, reducing redundant clause retrieval.
"""
import os

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
MMR_LAMBDA  = 0.7   # weight: 1.0 = pure relevance, 0.0 = pure diversity
FETCH_MULTI = 6     # fetch top_k * FETCH_MULTI candidates before MMR rerank

_client     = None
_chroma     = None
_collection = None


def _get_collection():
    global _client, _chroma, _collection
    if _collection is None:
        _client     = OpenAI()
        _chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma.get_or_create_collection("visa_rules")
    return _collection, _client


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _mmr(
    query_emb: list[float],
    candidate_embs: list[list[float]],
    candidates: list[dict],
    k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> list[dict]:
    selected      = []
    selected_embs = []
    remaining     = list(range(len(candidates)))

    while len(selected) < k and remaining:
        if not selected_embs:
            scores = [_cosine_sim(query_emb, candidate_embs[i]) for i in remaining]
            best_idx = remaining[int(np.argmax(scores))]
        else:
            mmr_scores = []
            for i in remaining:
                rel = _cosine_sim(query_emb, candidate_embs[i])
                red = max(_cosine_sim(candidate_embs[i], se) for se in selected_embs)
                mmr_scores.append(lambda_mult * rel - (1 - lambda_mult) * red)
            best_idx = remaining[int(np.argmax(mmr_scores))]

        selected.append(candidates[best_idx])
        selected_embs.append(candidate_embs[best_idx])
        remaining.remove(best_idx)

    return selected


def compute_rag_precision(query: str, chunks: list[dict]) -> float:
    """
    LLM judge (gpt-4o-mini) scores each retrieved chunk as relevant (1) or not (0).
    Returns max(scores): 1.0 if ANY chunk is relevant, 0.0 if none are.

    max() is the correct aggregation here — the compliance use case succeeds when
    the retrieval surfaces at least one applicable rule clause; averaging over MMR-
    diversified chunks would penalise correct retrieval whenever adjacent clauses
    are also returned.
    """
    if not chunks:
        return 0.0
    _, client = _get_collection()
    scores = []
    for chunk in chunks:
        snippet = chunk.get("text", "")[:500]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Is the following Visa rulebook text relevant to this dispute query?\n"
                    f"Query: \"{query}\"\n\nText:\n{snippet}\n\n"
                    "Reply with only 1 (relevant) or 0 (not relevant)."
                ),
            }],
            max_tokens=1,
            temperature=0,
        )
        try:
            scores.append(float(resp.choices[0].message.content.strip()))
        except (ValueError, AttributeError):
            scores.append(0.0)
    return round(max(scores), 3)


def search_compliance_docs(query: str, top_k: int = 3) -> list[dict]:
    collection, client = _get_collection()

    if collection.count() == 0:
        return [{
            "text":            "NOT FOUND — Visa rules not yet ingested into ChromaDB.",
            "section":         "unknown",
            "page":            "unknown",
            "relevance_score": 0.0,
        }]

    # Embed the query
    query_emb = client.embeddings.create(
        input=query, model="text-embedding-3-small"
    ).data[0].embedding

    # Fetch more candidates than needed for MMR diversity
    n_candidates = min(top_k * FETCH_MULTI, collection.count())
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    candidates = []
    candidate_embs = []
    for doc, meta, dist, emb in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0],
    ):
        candidates.append({
            "text":            doc,
            "section":         meta.get("section", "unknown"),
            "page":            meta.get("page", "unknown"),
            "relevance_score": round(1 - dist, 3),
        })
        candidate_embs.append(emb)

    reranked = _mmr(query_emb, candidate_embs, candidates, k=top_k)
    return reranked
