import subprocess
import time
import requests


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL = "llama3.2:latest"


_ollama_process = None


def is_ollama_running() -> bool:
    """
    Checks if the local Ollama server is available.
    """

    try:
        response = requests.get(
            OLLAMA_BASE_URL,
            timeout=2
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def start_ollama_server() -> bool:
    """
    Starts Ollama server locally if it is not already running.

    Returns True if Ollama is available after the start attempt.
    """

    global _ollama_process

    if is_ollama_running():
        return True

    try:
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False
        )

    except FileNotFoundError:
        return False

    except Exception:
        return False

    # Give Ollama a few seconds to start.
    for _ in range(10):
        time.sleep(0.5)

        if is_ollama_running():
            return True

    return False


def ask_ollama(
    user_message: str,
    conversation_history: list[dict] | None = None
) -> str:
    """
    Sends a message to the local Ollama model and returns the assistant response.
    """

    if conversation_history is None:
        conversation_history = []

    ollama_ready = start_ollama_server()

    if not ollama_ready:
        return (
            "Ollama n'est pas disponible. Vérifiez qu'Ollama est installé "
            "et que la commande ollama fonctionne dans PowerShell."
        )

    system_prompt = {
        "role": "system",
        "content": (
            "Tu es un assistant vocal intégré à un dashboard Streamlit. "
            "Tu aides l'utilisateur à comprendre les données affichées. "
            "Réponds en français, clairement, avec des réponses courtes. "
            "La réponse sera lue à voix haute, donc évite les tableaux, "
            "les longues listes et les réponses trop longues."
        )
    }

    messages = [system_prompt] + conversation_history + [
        {
            "role": "user",
            "content": user_message
        }
    ]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=90
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"].strip()

    except Exception as error:
        return (
            "Erreur lors de l'appel à Ollama. "
            f"Détail technique : {error}"
        )