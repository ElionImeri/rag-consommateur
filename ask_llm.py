import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:latest"


def ask(question: str) -> str:
    # Send a question to the model and returns its response.
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


if __name__ == "__main__":
    questions = [
        "Un commerçant peut-il refuser de réparer un appareil encore sous garantie au Québec?",
        "In Quebec, can I cancel a contract signed with door-to-door salesperson?",
    ]
    for q in questions:
        print("QUESTION : ", q)
        print("RESPONSE : ", ask(q))
        print("-" * 60)