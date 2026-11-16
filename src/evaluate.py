import json
import time
from pathlib import Path

from search import search
from stats_corpus import load_articles

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
RESULTS_PATH = Path("data/eval/results_section.jsonl")
K = 5  # number of distinct article
N_CHUNKS = 20


def top_articles(question, k=K):
    """Return the first distinct article numbers, in ranking order."""
    articles = []
    for chunk, score in search(question, k=N_CHUNKS):
        if chunk["article"] not in articles:
            articles.append(chunk["article"])
        if len(articles) == k:
            break
    return articles


def reciprocal_rank(retrieved, expected):
    """1 / rank if the first expected article or 0 if none is found."""
    for rank, number in enumerate(retrieved, start=1):
        if number in expected:
            return 1 / rank
    return 0.0


if __name__ == "__main__":
    questions = [q for q in load_articles(QUESTIONS_PATH) if q["articles"]]
    print(f"{len(questions)} questions with expected articles")

    results = []
    start = time.perf_counter()
    for q in questions:
        retrieved = top_articles(q["question"])
        hit = any(a in retrieved for a in q["articles"])
        rr = reciprocal_rank(retrieved, q["articles"])
        results.append({**q, "retrieved": retrieved, "hit": hit, "rr": rr})
    elapsed = time.perf_counter() - start

    hit_at_k = sum(r["hit"] for r in results) / len(results)
    mrr = sum(r["rr"] for r in results) / len(results)
    print(f"hit@{K}: {hit_at_k:.1%}")
    print(f"MRR: {mrr:.3f}")
    print(f"Average latency: {elapsed / len(results) * 1000:.0f} ms per question")

    # Save the details to analyse failures
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nMisses:")
    for r in results:
        if not r["hit"]:
            print(f"{r['id']}: expected {r['articles']}, got {r['retrieved']}")
