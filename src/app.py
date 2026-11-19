from fastapi import FastAPI
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="Assistant LPC", description="Quebec Consumer Protection Act RAG")

class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    article: str
    section: str
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.get("/health")
def health():
    """Simple check that the service is up."""
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question about the Consumer Protection Act, with sources."""
    reply, chunks = answer(request.question)
    sources = [
        Source(article=c["article"], section=c["section"], text=c["text"])
        for c in chunks
    ]
    return AskResponse(answer=reply, sources=sources)