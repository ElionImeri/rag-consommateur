import json
import re
from pathlib import Path

from stats_corpus import load_articles

LIMIT = 2000  # 2000 characters

ITEM_MARKERS = {
    "fr": re.compile(r"\s(?=(?:[a-z](?:\.\d+)*\)|\d+(?:\.\d+)*°))"),
    "en": re.compile(r"\s(?=(?:\([a-z](?:\.\d+)*\)|\(\d+(?:\.\d+)*\)))"),
}


def split_items(text, lang):
    # Split an article into its lead-in sentence (chapeau) and its sub-points.
    parts = ITEM_MARKERS[lang].split(text)
    chapeau = parts[0].strip()
    items = [p.strip() for p in parts[1:]]
    return chapeau, items


def chunk_article(article):
    # Return  a list of chunks (dicts) for one article.
    text = article["text"]

    # For articles under the LIMIT, keep them untouched
    if len(text) <= LIMIT:
        return [make_chunk(article, 0, text)]

    chapeau, items = split_items(text, article["lang"])

    if not items:
        sentences = re.split(r"(?<=\D[.!?])\s+", text.strip())
        chapeau = sentences[0]
        items = sentences[1:]

    chunks = []
    current = chapeau
    for item in items:
        if len(current) + 1 + len(item) > LIMIT:
            if current != chapeau:
                chunks.append(
                    make_chunk(article, len(chunks), current)
                )  # save the chunk and start a new one
            current = chapeau + " " + item
        else:
            current += " " + item  # Add the sub points to the chunk
    chunks.append(make_chunk(article, len(chunks), current))
    return chunks


def make_chunk(article, part, text):
    # Build a chunk dict from an article, its part number and its text.
    return {
        "id": f"{article['id']}#{part}",
        "article": article["article"],
        "lang": article["lang"],
        "part": part,
        "text": text,
    }


if __name__ == "__main__":
    for lang in ["fr", "en"]:
        in_path = Path(f"data/processed/articles_{lang}.jsonl")
        out_path = Path(f"data/processed/chunks_{lang}.jsonl")

        articles = load_articles(in_path)

        all_chunks = []
        for article in articles:
            all_chunks.extend(chunk_article(article))

        print(f"[{lang}] {len(articles)} articles -> {len(all_chunks)} chunks")
        print(
            f"[{lang}] longest chunk: {max(len(c['text']) for c in all_chunks)} characters"
        )

        with open(out_path, "w", encoding="utf-8") as fh:
            for chunk in all_chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"[{lang}], saved in {out_path}")

        too_long = [c for c in all_chunks if len(c["text"]) > LIMIT]
        print(f"[{lang}] {len(too_long)} chunks over the limit:")
        for c in too_long:
            print(f"{c['id']}: {len(c['text'])} characters")
