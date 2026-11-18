import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:latest"

def chat(messages, temperature=0.1, num_ctx=8192):
    """Send a list of messages to the local model and return its answer."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except:
        raise SystemExit("Cannot reach Ollama on localhost:11434. Is the server running?")
    response.raise_for_status()
    return response.json()["message"]["content"]