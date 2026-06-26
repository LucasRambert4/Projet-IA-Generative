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
            timeout=2,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def start_ollama_server() -> bool:
    """
    Starts Ollama locally if it is not already running.

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
            shell=False,
        )

    except FileNotFoundError:
        return False

    except Exception:
        return False

    for _ in range(20):
        time.sleep(0.5)

        if is_ollama_running():
            return True

    return False


def build_system_prompt(dashboard_context: str | None = None) -> dict:
    """
    Builds the system prompt given to Ollama.

    The dashboard context is injected here so the model answers only about
    the local Streamlit dashboard data.
    """

    if not dashboard_context:
        dashboard_context = "Aucun contexte dashboard fourni."

    return {
        "role": "system",
        "content": (
            "Tu es un assistant vocal intégré à un dashboard Streamlit local de démonstration. "
            "Tu analyses uniquement les données fournies dans le contexte du dashboard. "
            "Tu ne dois jamais parler du site officiel Streamlit, du trafic web réel, "
            "de données temps réel ou de données externes. "
            "Tu dois utiliser les données fournies pour calculer les moyennes, totaux, maximums "
            "et comparaisons lorsque c'est possible. "
            "Tu ne dois pas dire qu'une donnée est absente si elle peut être calculée avec le contexte. "
            "Ne dis jamais à l’utilisateur qu’il a déjà posé la question. Même si la question est répétée, réponds normalement. "
            "Si l'utilisateur parle de visiteurs, comprends-le comme les clients du dashboard, "
            "sauf indication contraire. "
            "Réponds en français. "
            "Fais des réponses courtes, naturelles, utiles et faciles à lire à voix haute. "
            "Évite les tableaux Markdown et les longues listes. "
            "Réponds en une ou deux phrases maximum.\n\n"
            f"CONTEXTE DU DASHBOARD :\n{dashboard_context}"
        ),
    }


def ask_ollama(
    user_message: str,
    conversation_history: list[dict] | None = None,
    dashboard_context: str | None = None,
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

    system_prompt = build_system_prompt(
        dashboard_context=dashboard_context,
    )

    messages = [system_prompt] + conversation_history + [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_ctx": 1536,
            "num_predict": 90,
            "repeat_penalty": 1.1,
        },
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=75,
        )

        response.raise_for_status()

        data = response.json()

        assistant_message = data.get("message", {})
        assistant_content = assistant_message.get("content", "")

        if not assistant_content:
            return "Ollama a répondu, mais aucune réponse textuelle n'a été reçue."

        return assistant_content.strip()

    except requests.exceptions.Timeout:
        return (
            "Ollama met trop de temps à répondre. "
            "La question est peut-être trop longue ou le modèle est encore en chargement."
        )

    except Exception as error:
        return (
            "Erreur lors de l'appel à Ollama. "
            f"Détail technique : {error}"
        )