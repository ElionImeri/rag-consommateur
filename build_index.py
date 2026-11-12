import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from stats_corpus import load_articles

MODEL_NAME = "intfloat/multilingual-e5-base"
OUT_DIR = Path("data/index")

if __name__ == "__main__":
    # 1. Load the chunks of both languages into a single list
    chunks = []
    for lang in ["fr", "en"]:
        chunks.extend(load_articles(Path(f"data/processed/chunks_{lang}.jsonl")))
    print(f"{len(chunks)} chunks loaded")

    model = SentenceTransformer(MODEL_NAME)

    # 2. Check the real length in tokens (the model's limit is 512)
    token_counts = [len(model.tokenizer.tokenize(chunk["text"])) for chunk in chunks]
    if max(token_counts) + 5 > 512:
       raise SystemExit("A chunk is too long for the model (512 tokens): lower LIMIT.")
    print(f"Longest chunk: {max(token_counts)} tokens")

    # 3. Encode every chunk, with "passage:" prefix
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    print(embeddings.shape)

    # 4. Save the matrix and the chunks in the same order
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "embeddings.npy", embeddings)
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
