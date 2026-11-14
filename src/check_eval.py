from pathlib import Path
from stats_corpus import load_articles

ARTICLES_PATH = Path("data/processed/articles_fr.jsonl")
QUESTIONS_PATH = Path("data/eval/questions.jsonl")

if __name__ == "__main__":
    articles = load_articles(ARTICLES_PATH)
    questions = load_articles(QUESTIONS_PATH)

    # 1. All articles numbers that exist in the corpus
    existing = {a["article"] for a in articles}

    # 2. Check every labelled article of every question
    missing = []
    for q in questions:
        for number in q["articles"]:
            if number not in existing:
                missing.append((q["id"], number))

    # 3. Report
    print(f"{len(questions)} questions checked")
    print(
        f"{sum(1 for q in questions if not q['articles'])} questions without expected article"
    )

    if missing:
        print(f"{len(missing)} labelled articles NOT found in the corpus:")
        for qid, number in missing:
            print(f"    {qid}: article {number}")
    else:
        print("All labelled articles exists in the corpus")
