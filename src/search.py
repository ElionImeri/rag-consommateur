from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from stats_corpus import load_articles

MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_DIR = Path("data/index")
RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
reranker = CrossEncoder(RERANKER_NAME, max_length=512)

# Loaded once, when the module is imported
model = SentenceTransformer(MODEL_NAME)
embeddings = np.load(INDEX_DIR / "embeddings.npy")
chunks = load_articles(INDEX_DIR / "chunks.jsonl")


def search(question, k=5):
    """Return the k chunks closest to the question, with their scores."""
    q_vec = model.encode(f"query: {question}", normalize_embeddings=True)
    scores = embeddings @ q_vec
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


def search_rerank(question, k=5, n_candidates=60):
    """Retrieve n_candidates chunks with embeddings, the rerank them
    with cross-encoder and return the k best, with their new scores."""
    candidates = search(question, k=n_candidates)
    pairs = [(question, f"{c['section']}\n{c['text']}") for c, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip([c for c, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]


if __name__ == "__main__":
    questions = [
        "Combien de jours ai-je pour annuler un contrat avec un vendeur itinérant ?",
        "How many days do I have to cancel a contract with a door-to-door salesperson?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        for chunk, score in search(q):
            print(f" {score:.3f} {chunk['id']} {chunk['text'][:80]}")
