from llm import chat
from search import search
from langdetect import DetectorFactory, LangDetectException, detect

SYSTEM_PROMPT = """You are an assistant that provides information about Quebec's Consumer Protection Act. Your task is to answer exclusively based on the provided articles (never on general knowledge).
You must cite the relevant article after every statement, using the following format: [Article 59]. For example: The consumer may cancel the contract within 10 days [Article 59].
The articles are taken from Quebec's Consumer Protection Act (CQLR, chapter P-40.1). Do not cite any other legislation.
If the provided articles do not contain enough information to answer the question, state this clearly. Do not make up an answer.
Always respond in the same language as the question.
Keep your answers concise. Write no more than a few paragraphs.
Legal Disclaimer: The information provided is for general informational and educational purposes only. It is based exclusively on the articles of Quebec's Consumer Protection Act (CQLR, c. P-40.1) provided as sources. It does not constitute legal advice, a legal opinion, or a substitute for consulting a qualified legal professional. The applicability of the law may depend on the specific circumstances of each situation.
Some of the provided articles may not apply to the user's situation. Use only the articles that clearly apply and ignore the others"""

DetectorFactory.seed = 0
LANG_NAME = {"fr": "French", "en": "English"}
DEFAULT_LANG = "fr"
DISCLAIMER = {
    "fr": "Ceci est de l'information juridique et non un avis juridique.",
    "en": "This is legal information, not legal advice.",
}


def detect_lang(text):
    """Return 'fr' or 'en'; fallback on DEFAULT_LANG if unsure."""
    try:
        lang = detect(text)
    except LangDetectException:
        return DEFAULT_LANG
    if lang in LANG_NAME:
        return lang
    return DEFAULT_LANG


def retrieve(question, k=5, n_chunks=20):
    """Return the first chunk of the k most relevant distinct articles."""
    chunks = []
    seen = set()
    for chunk, score in search(question, k=n_chunks):
        if chunk["article"] not in seen:
            seen.add(chunk["article"])
            chunks.append(chunk)
        if len(chunks) == k:
            break
    return chunks


def build_context(chunks):
    """Format the retrieved chunks so the model can cite them."""
    return "\n\n".join(
        f"[Article {c['article']}] ({c['section']})\n{c['text']}" for c in chunks
    )


def answer(question):
    """Answer a question from the law, and return the answer and its source."""
    lang = detect_lang(question)
    chunks = retrieve(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Article: \n\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer in {LANG_NAME[lang]}",
        },
    ]
    reply = chat(messages)
    reply = reply.strip() + "\n\n" + DISCLAIMER[lang]
    return reply, chunks


if __name__ == "__main__":
    questions = [
        "Un commerçant peut-il refuser de réparer un appareil encore sous garantie au Québec ?",
        "In Quebec, can I cancel a contract signed with a door-to-door salesperson?",
    ]
    for q in questions:
        reply, sources = answer(q)
        print(f"QUESTION: {q}\n")
        print(reply)
        print("\nSources :", [c["article"] for c in sources])
        print("-" * 60)
