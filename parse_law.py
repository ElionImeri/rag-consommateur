import re
from pathlib import Path
import json

from bs4 import BeautifulSoup

HTML_FILE = Path("data/tmp/epub_fr/OPF/page2.xhtml")
EPUB_DIR = Path("data/tmp/epub_fr/OPF")
files = EPUB_DIR.glob("page*.xhtml")


def parse_file(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    for h in soup.find_all(class_="Hidden"):
        # print(len(soup.find_all("Hidden"))) would have given 0 because Hidden is not an element like <span></span> or <head></head>
        h.decompose()

    for n in soup.select(
        "[style*='display: none']"
    ):  # without any element like <span></span> it will apply to all element that contain display: none
        n.decompose()

    # To delete all historic foot notes in the article
    for subdiv in soup.select("[style*='font-size: 9pt']"):
        subdiv.decompose()

    ARTICLE_ID = re.compile(r"^se:[^-]+$")  # This

    articles = []
    for div in soup.find_all("div", id=ARTICLE_ID):
        text = div.get_text(" ", strip=True)
        number = div["id"].removeprefix("se:").replace("_", ".")
        articles.append(
            {
                "id": f"fr-{number}",
                "article": number,
                "lang": "fr",
                "text": text,
            }
        )
    return articles


if __name__ == "__main__":
    files = sorted(
        EPUB_DIR.glob("page*.xhtml"),
        key=lambda f: int(re.search(r"\d+", f.stem).group()),
    )

    all_articles = []
    for f in files:
        all_articles.extend(parse_file(f))

    print("Number of articles: ", len(all_articles))
    print("First: ", all_articles[0]["article"])
    print("Last: ", all_articles[-1]["article"])
    print("Unique IDs :", len({a["id"] for a in all_articles}))

    out_path = Path("data/processed/articles_fr.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for article in all_articles:
            line = json.dumps(article, ensure_ascii=False)
            fh.write(line + "\n")

    print("Saved in", out_path)
