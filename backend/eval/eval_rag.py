"""
RAG retrieval evaluation — runs the 20-query golden set through
search_compliance_docs and measures Hit@1, Hit@3, and MRR.
"""

import numpy as np

from backend.eval.rag_golden_set import RAG_GOLDEN_SET
from backend.tools.rag_tool import search_compliance_docs


def _norm(text: str) -> str:
    """Collapse PDF whitespace artifacts: extra spaces, hard line breaks."""
    return " ".join(text.split()).lower()


def eval_rag() -> dict:
    hit_at_1  = []
    hit_at_3  = []
    rr_scores = []
    failed_queries = []

    for item in RAG_GOLDEN_SET:
        results = search_compliance_docs(item["query"], top_k=3)

        chunk_scores = []
        for chunk in results:
            fragment_found = _norm(item["expected_text_fragment"]) in _norm(chunk["text"])
            section_found  = item["expected_section"] in chunk.get("section", "")
            chunk_scores.append(1.0 if (fragment_found or section_found) else 0.0)

        h1 = chunk_scores[0] == 1.0 if chunk_scores else False
        h3 = max(chunk_scores) == 1.0 if chunk_scores else False

        rr = 0.0
        for rank, score in enumerate(chunk_scores, start=1):
            if score == 1.0:
                rr = 1.0 / rank
                break

        hit_at_1.append(int(h1))
        hit_at_3.append(int(h3))
        rr_scores.append(rr)

        if not h3:
            failed_queries.append({
                "id":               item["id"],
                "query":            item["query"],
                "expected_section": item["expected_section"],
                "top_result":       results[0]["text"][:120] if results else "NO RESULTS",
            })

    hit1 = round(float(np.mean(hit_at_1)), 3)
    hit3 = round(float(np.mean(hit_at_3)), 3)
    mrr  = round(float(np.mean(rr_scores)), 3)

    print("\n── RAG RETRIEVAL EVALUATION ────────────────────────")
    print(f"Hit@1:  {hit1:.3f}  ({sum(hit_at_1)}/{len(hit_at_1)} queries)")
    print(f"Hit@3:  {hit3:.3f}  ({sum(hit_at_3)}/{len(hit_at_3)} queries)")
    print(f"MRR:    {mrr:.3f}")

    if failed_queries:
        print(f"\nFailed queries ({len(failed_queries)}):")
        for fq in failed_queries:
            print(f"  [{fq['id']}] {fq['query']}")
            print(f"    expected: §{fq['expected_section']}")
            print(f"    got:      {fq['top_result']}")

    return {
        "hit_at_1":       hit1,
        "hit_at_3":       hit3,
        "mrr":            mrr,
        "n_queries":      len(RAG_GOLDEN_SET),
        "failed_queries": failed_queries,
    }
