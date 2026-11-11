import json
from pathlib import Path

IN_PATH = Path("data/processed/articles_fr.jsonl")
LIMIT = 2000  # number of characters (about 512 tokens)


def load_articles(path):
    # Read a JSON file and return a list of dicts.
    articles = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            articles.append(json.loads(line))  # transform a json line into a dict
    return articles


if __name__ == "__main__":
    articles = load_articles(IN_PATH)
    lengths = [len(a["text"]) for a in articles]

    average_length = sum(lengths) / len(lengths)
    print(average_length)

    # find the 10 longest articles
    longest = sorted(articles, key=lambda a: len(a["text"]), reverse=True)
    for i, article in enumerate(longest[:10], start=1):
        print(f"{i}. Article {article['article']} : {len(article['text'])} characters")

    count = 0
    # Show every article that surpasses the LIMIT
    for i, article in enumerate(articles, start=1):
        length = len(article["text"])
        if length > LIMIT:
            count += 1
            print(f"Article {article['article']}: {length} characters")

    print(f"{count} articles over {LIMIT} characters")
