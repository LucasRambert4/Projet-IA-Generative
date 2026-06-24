import base64
import os
import subprocess
import sys
import time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ==========================================================
# PROJECT PATH
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
LISTENER_PID_FILE = RUNTIME_DIR / "live_listener.pid"
LISTENER_LOG_FILE = RUNTIME_DIR / "live_listener.log"
LISTENER_SCRIPT = ROOT_DIR / "src" / "continuous_listener_live.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.command_bus import read_latest_command_event
from src.dashboard_controller import (
    initialize_dashboard_state,
    apply_dashboard_command,
    get_page_label,
)


try:
    from src.ollama_client import ask_ollama, start_ollama_server
except Exception:
    ask_ollama = None
    start_ollama_server = None


try:
    from src.kokoro_tts_engine import synthesize_response_to_wav
except Exception:
    synthesize_response_to_wav = None


# ==========================================================
# STREAMLIT CONFIG
# ==========================================================

st.set_page_config(
    page_title="Voice Dashboard",
    page_icon="🎙️",
    layout="wide"
)


# ==========================================================
# SESSION STATE
# ==========================================================

initialize_dashboard_state(st)

if "chatbot_open" not in st.session_state:
    st.session_state.chatbot_open = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_tts_audio" not in st.session_state:
    st.session_state.last_tts_audio = None

if "last_processed_event_id" not in st.session_state:
    st.session_state.last_processed_event_id = None

if "last_scroll_action_id" not in st.session_state:
    st.session_state.last_scroll_action_id = 0

if "ollama_checked" not in st.session_state:
    st.session_state.ollama_checked = False

if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = False

if "listener_checked" not in st.session_state:
    st.session_state.listener_checked = False

if "listener_available" not in st.session_state:
    st.session_state.listener_available = False

if "listener_status" not in st.session_state:
    st.session_state.listener_status = "Live listener non vérifié."


# ==========================================================
# START OLLAMA WHEN DASHBOARD STARTS
# ==========================================================

if not st.session_state.ollama_checked:
    if start_ollama_server is not None:
        st.session_state.ollama_available = start_ollama_server()
    else:
        st.session_state.ollama_available = False

    st.session_state.ollama_checked = True


# ==========================================================
# DATA
# ==========================================================

sales_by_month = pd.DataFrame(
    {
        "mois": ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin"],
        "ventes": [12000, 15000, 17000, 14000, 21000, 26000],
        "clients": [120, 150, 180, 160, 210, 260],
    }
)

sales_by_region = pd.DataFrame(
    {
        "region": ["Nord", "Sud", "Est", "Ouest"],
        "ventes": [32000, 28000, 19000, 24000],
        "clients": [300, 240, 170, 210],
    }
)


# ==========================================================
# HELPERS
# ==========================================================

def is_windows_process_running(pid: int) -> bool:
    """
    Checks if a Windows process is still running.
    """

    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        return str(pid) in result.stdout

    except Exception:
        return False


def is_listener_already_running() -> bool:
    """
    Checks if the live listener process from the PID file is still running.
    """

    if not LISTENER_PID_FILE.exists():
        return False

    try:
        pid = int(LISTENER_PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return False

    return is_windows_process_running(pid)


def start_live_listener_if_needed() -> tuple[bool, str]:
    """
    Starts continuous_listener_live.py automatically with the dashboard.

    Returns:
    - success boolean
    - status message
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    if is_listener_already_running():
        return True, "Live listener déjà actif."

    if not LISTENER_SCRIPT.exists():
        return False, "Fichier src/continuous_listener_live.py introuvable."

    try:
        log_file = open(LISTENER_LOG_FILE, "a", encoding="utf-8")

        creation_flags = 0

        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            [sys.executable, str(LISTENER_SCRIPT)],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        LISTENER_PID_FILE.write_text(
            str(process.pid),
            encoding="utf-8"
        )

        st.session_state.listener_process = process

        return True, f"Live listener démarré automatiquement. PID: {process.pid}"

    except Exception as error:
        return False, f"Impossible de démarrer le live listener : {error}"


def run_scroll_if_needed():
    """
    Executes pending scroll action injected by dashboard_controller.py.
    """

    current_scroll_id = st.session_state.get("scroll_action_id", 0)
    last_scroll_id = st.session_state.get("last_scroll_action_id", 0)

    if current_scroll_id == last_scroll_id:
        return

    direction = st.session_state.get("pending_scroll_direction")
    amount = st.session_state.get("pending_scroll_amount", 700)

    if not direction:
        return

    if direction == "down":
        script = f"window.parent.scrollBy({{top: {amount}, left: 0, behavior: 'smooth'}});"
    elif direction == "up":
        script = f"window.parent.scrollBy({{top: -{amount}, left: 0, behavior: 'smooth'}});"
    elif direction == "top":
        script = "window.parent.scrollTo({top: 0, left: 0, behavior: 'smooth'});"
    elif direction == "bottom":
        script = "window.parent.scrollTo({top: document.body.scrollHeight, left: 0, behavior: 'smooth'});"
    else:
        return

    components.html(
        f"""
        <script>
        {script}
        </script>
        """,
        height=0,
    )

    st.session_state.last_scroll_action_id = current_scroll_id



def ask_chatbot(user_message: str) -> str:
    """
    Sends a message to Ollama and returns the assistant response.
    """

    if ask_ollama is None:
        st.session_state.ollama_available = False
        return (
            "Ollama n'est pas encore connecté dans l'application. "
            "Vérifiez le fichier src/ollama_client.py."
        )

    conversation_history = st.session_state.chat_history[-8:]

    try:
        response = ask_ollama(
            user_message=user_message,
            conversation_history=conversation_history,
        )

        st.session_state.ollama_available = True

        return response

    except Exception:
        st.session_state.ollama_available = False

        return (
            "Ollama n'est pas disponible pour le moment. "
            "Vérifiez que le modèle llama3.2:latest est installé."
        )

# ==========================================================
# START LIVE LISTENER WHEN DASHBOARD STARTS
# ==========================================================

if not st.session_state.listener_checked:
    listener_available, listener_status = start_live_listener_if_needed()

    st.session_state.listener_available = listener_available
    st.session_state.listener_status = listener_status
    st.session_state.listener_checked = True


def generate_tts_audio(text: str) -> str | None:
    """
    Converts assistant text response to audio using Kokoro TTS.
    """

    if synthesize_response_to_wav is None:
        return None

    try:
        return synthesize_response_to_wav(text)

    except NotImplementedError:
        return None

    except Exception as error:
        st.warning(f"Erreur Kokoro TTS : {error}")
        return None


def process_chatbot_message(user_message: str):
    """
    Processes a user message in chatbot mode.
    The popup stays open after the assistant response.
    """

    if not user_message:
        return

    # The visual chatbot stays open after a question.
    st.session_state.chatbot_open = True

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    assistant_response = ask_chatbot(user_message)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )

    audio_path = generate_tts_audio(assistant_response)
    st.session_state.last_tts_audio = audio_path

    st.session_state.status_message = "Réponse chatbot générée."

def process_voice_command(command: dict):
    """
    Routes a parsed voice command:
    - dashboard command
    - chatbot command
    - close chatbot
    """

    intent = command.get("intent")

    if intent == "open_chatbot":
        st.session_state.chatbot_open = True
        st.session_state.status_message = "Chatbot ouvert. Posez votre question."
        return

    if intent == "close_chatbot":
        st.session_state.chatbot_open = False
        st.session_state.last_tts_audio = None
        st.session_state.status_message = "Chatbot fermé."
        return

    if intent == "chatbot_message":
        message = command.get("message", "")
        process_chatbot_message(message)
        return

    apply_dashboard_command(st, command)


def read_and_process_latest_voice_event():
    """
    Reads latest command from runtime/latest_command.json
    and processes it only once.
    """

    event = read_latest_command_event()

    if not event:
        return

    event_id = event.get("event_id")

    if not event_id:
        return

    if event_id == st.session_state.last_processed_event_id:
        return

    st.session_state.last_processed_event_id = event_id

    st.session_state.last_transcription = event.get("transcription")
    st.session_state.last_wake_result = event.get("wake_result")

    parsed_command = event.get("parsed_command", {})

    process_voice_command(parsed_command)


def get_audio_html(audio_path: str | None) -> str:
    """
    Creates an HTML audio player with optional autoplay.
    """

    if not audio_path:
        return ""

    path = Path(audio_path)

    if not path.exists():
        return ""

    try:
        audio_bytes = path.read_bytes()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return f"""
        <audio controls autoplay style="width: 100%; margin-top: 10px;">
            <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
        </audio>
        """

    except Exception:
        return ""


def render_html(html_content: str):
    """
    Renders HTML safely in Streamlit.

    st.html is preferred because it avoids raw HTML appearing as visible text.
    """

    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def render_chatbot_popup():
    """
    Renders the floating chatbot button or a WhatsApp-like chat popup.
    """

    css_html = """
    <style>
    .chatbot-floating-button {
        position: fixed;
        right: 24px;
        bottom: 24px;
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: #111827;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        z-index: 9999;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .chatbot-popup {
        position: fixed;
        right: 24px;
        bottom: 100px;
        width: 390px;
        height: 540px;
        background: #f3f4f6;
        color: #111827;
        border-radius: 24px;
        z-index: 9999;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.24);
        border: 1px solid #e5e7eb;
        font-family: Arial, sans-serif;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }

    .chatbot-header {
        background: #111827;
        color: white;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .chatbot-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #22c55e;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
        flex-shrink: 0;
    }

    .chatbot-title-block {
        display: flex;
        flex-direction: column;
    }

    .chatbot-title {
        font-size: 16px;
        font-weight: 700;
        margin: 0;
    }

    .chatbot-status {
        font-size: 12px;
        color: #d1d5db;
        margin-top: 2px;
    }

    .chatbot-messages {
        flex: 1;
        padding: 16px;
        overflow-y: auto;
        background: linear-gradient(180deg, #f9fafb 0%, #eef2f7 100%);
    }

    .message-row {
        display: flex;
        margin-bottom: 12px;
        width: 100%;
    }

    .message-row.user-row {
        justify-content: flex-end;
    }

    .message-row.assistant-row {
        justify-content: flex-start;
    }

    .bubble {
        max-width: 78%;
        padding: 10px 13px;
        border-radius: 18px;
        font-size: 14px;
        line-height: 1.42;
        word-wrap: break-word;
        white-space: pre-wrap;
    }

    .bubble-user {
        background: #dcf8c6;
        color: #111827;
        border-bottom-right-radius: 5px;
        border: 1px solid #c8efb0;
    }

    .bubble-assistant {
        background: white;
        color: #111827;
        border-bottom-left-radius: 5px;
        border: 1px solid #e5e7eb;
    }

    .bubble-label {
        display: block;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 4px;
        opacity: 0.7;
    }

    .chatbot-footer {
        background: #ffffff;
        border-top: 1px solid #e5e7eb;
        padding: 12px 14px;
        font-size: 12px;
        color: #6b7280;
    }

    .chatbot-footer strong {
        color: #111827;
    }

    .chatbot-audio {
        margin-top: 10px;
    }
    </style>
    """

    render_html(css_html)

    if not st.session_state.chatbot_open:
        render_html(
            """
            <div class="chatbot-floating-button">💬</div>
            """
        )
        return

    messages_html = ""

    if not st.session_state.chat_history:
        messages_html = """
        <div class="message-row assistant-row">
            <div class="bubble bubble-assistant">
                <span class="bubble-label">Assistant</span>
                Chatbot ouvert. Posez une question à la voix.
            </div>
        </div>
        """
    else:
        for message in st.session_state.chat_history[-10:]:
            role = message.get("role", "assistant")
            content = escape(message.get("content", ""))

            if role == "user":
                messages_html += f"""
                <div class="message-row user-row">
                    <div class="bubble bubble-user">
                        <span class="bubble-label">Vous</span>
                        {content}
                    </div>
                </div>
                """
            else:
                messages_html += f"""
                <div class="message-row assistant-row">
                    <div class="bubble bubble-assistant">
                        <span class="bubble-label">Assistant</span>
                        {content}
                    </div>
                </div>
                """

    audio_html = get_audio_html(st.session_state.last_tts_audio)

    if audio_html:
        audio_html = f"""
        <div class="chatbot-audio">
            {audio_html}
        </div>
        """

    ollama_status = (
        "Ollama actif"
        if st.session_state.get("ollama_available")
        else "Ollama en démarrage ou indisponible"
    )

    chatbot_html = f"""
    <div class="chatbot-popup">
        <div class="chatbot-header">
            <div class="chatbot-avatar">🤖</div>
            <div class="chatbot-title-block">
                <div class="chatbot-title">Assistant vocal</div>
                <div class="chatbot-status">{ollama_status}</div>
            </div>
        </div>

        <div class="chatbot-messages">
            {messages_html}
            {audio_html}
        </div>

        <div class="chatbot-footer">
            Dites <strong>chatbot désactivé</strong> ou <strong>ferme chatbot</strong> pour quitter ce mode.
        </div>
    </div>
    """

    render_html(chatbot_html)



# ==========================================================
# PROCESS VOICE COMMANDS
# ==========================================================

read_and_process_latest_voice_event()
run_scroll_if_needed()


# ==========================================================
# UI
# ==========================================================

st.title("🎙️ Voice-Controlled Dashboard")

st.caption(
    "Dashboard contrôlé à la voix avec wake word, commandes vocales, scroll, "
    "et mode chatbot local via Ollama."
)

st.info(st.session_state.status_message)


# ==========================================================
# TOP NAVIGATION
# ==========================================================

nav_cols = st.columns(4)

with nav_cols[0]:
    if st.button("Résumé"):
        st.session_state.current_page = "resume"

with nav_cols[1]:
    if st.button("Ventes"):
        st.session_state.current_page = "ventes"

with nav_cols[2]:
    if st.button("Clients"):
        st.session_state.current_page = "clients"

with nav_cols[3]:
    if st.button("Régions"):
        st.session_state.current_page = "regions"


current_page = st.session_state.current_page

st.divider()

st.subheader(f"Page actuelle : {get_page_label(current_page)}")


# ==========================================================
# DASHBOARD CONTENT
# ==========================================================

if st.session_state.selected_metric or st.session_state.selected_dimension:
    st.success(
        f"Commande graphique active : "
        f"métrique={st.session_state.selected_metric}, "
        f"dimension={st.session_state.selected_dimension}"
    )


if current_page == "resume":
    col1, col2, col3 = st.columns(3)

    total_sales = int(sales_by_month["ventes"].sum())
    total_clients = int(sales_by_month["clients"].sum())
    average_sales = int(sales_by_month["ventes"].mean())

    col1.metric("Ventes totales", f"{total_sales:,} €")
    col2.metric("Clients", total_clients)
    col3.metric("Vente moyenne", f"{average_sales:,} €")

    st.markdown("### Évolution des ventes")
    st.line_chart(
        sales_by_month.set_index("mois")["ventes"]
    )

    st.markdown("### Ventes par région")
    st.bar_chart(
        sales_by_region.set_index("region")["ventes"]
    )


elif current_page == "ventes":
    st.markdown("### Analyse des ventes")

    st.line_chart(
        sales_by_month.set_index("mois")["ventes"]
    )

    st.markdown("### Ventes par région")
    st.bar_chart(
        sales_by_region.set_index("region")["ventes"]
    )

    st.dataframe(sales_by_month, use_container_width=True)


elif current_page == "clients":
    st.markdown("### Analyse des clients")

    st.line_chart(
        sales_by_month.set_index("mois")["clients"]
    )

    st.markdown("### Clients par région")
    st.bar_chart(
        sales_by_region.set_index("region")["clients"]
    )

    st.dataframe(sales_by_region, use_container_width=True)


elif current_page == "regions":
    st.markdown("### Analyse régionale")

    st.bar_chart(
        sales_by_region.set_index("region")[["ventes", "clients"]]
    )

    st.dataframe(sales_by_region, use_container_width=True)


# Extra content to test voice scroll
st.divider()
st.markdown("## Zone de test du scroll vocal")

for index in range(1, 9):
    st.markdown(f"### Section {index}")
    st.write(
        "Cette section sert à tester les commandes vocales comme "
        "'descends', 'monte', 'tout en bas' ou 'retourne en haut'."
    )


# ==========================================================
# DEBUG PANEL
# ==========================================================

with st.expander("Debug vocal"):
    st.write("Dernière transcription :")
    st.code(st.session_state.get("last_transcription"))

    st.write("Dernière commande :")
    st.json(st.session_state.get("last_command"))

    st.write("Chatbot ouvert :")
    st.write(st.session_state.chatbot_open)

    st.write("Historique chatbot :")
    st.json(st.session_state.chat_history)

    st.write("Ollama disponible :")
    st.write(st.session_state.ollama_available)

    st.write("Live listener disponible :")
    st.write(st.session_state.listener_available)

    st.write("Statut live listener :")
    st.write(st.session_state.listener_status)

    st.write("Log listener :")
    st.code(str(LISTENER_LOG_FILE))


# ==========================================================
# CHATBOT POPUP
# ==========================================================

render_chatbot_popup()


# ==========================================================
# AUTO REFRESH
# ==========================================================
# The dashboard reruns regularly so it can read commands written
# by the voice listener in runtime/latest_command.json.

time.sleep(0.8)
st.rerun()