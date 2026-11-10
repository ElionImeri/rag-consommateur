import re
from pathlib import Path

from bs4 import BeautifulSoup

HTML_FILE = Path("data/tmp/epub_fr/OPF/page2.xhtml")

soup = BeautifulSoup(HTML_FILE.read_text(encoding="utf-8"), "html.parser")

for h in soup.find_all(class_="Hidden"):
    #print(len(soup.find_all("Hidden"))) would have given 0 because Hidden is not an element like <span></span> or <head></head>
    h.decompose()

for n in soup.select("[style*='display: none']"): # without any element like <span></span> it will apply to all element that contain display: none
    n.decompose()

# To delete all historic foot notes in the article
for subdiv in soup.select("[style*='font-size: 9pt']"):
        subdiv.decompose()

ARTICLE_ID = re.compile(r"^se:[^-]+$")

for div in soup.find_all("div", id=ARTICLE_ID):
    if div["id"] != "se:59":
        continue
    text = div.get_text(" ", strip=True)
    print(text)