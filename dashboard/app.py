import base64
import re
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh


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


from src.command_bus import read_latest_command_event, read_listener_status
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
    layout="wide",
)


@st.cache_resource
def get_chatbot_executor():
    """
    Creates one background worker for Ollama calls.
    """

    return ThreadPoolExecutor(max_workers=1)


@st.cache_resource
def get_tts_executor():
    """
    Creates one background worker for Kokoro TTS.

    Kokoro is separated from Ollama so audio generation does not slow down
    the next chatbot response.
    """

    return ThreadPoolExecutor(max_workers=1)


# ==========================================================
# SESSION STATE
# ==========================================================

initialize_dashboard_state(st)

if "chatbot_open" not in st.session_state:
    st.session_state.chatbot_open = False

if "chatbot_future" not in st.session_state:
    st.session_state.chatbot_future = None

if "chatbot_waiting_response" not in st.session_state:
    st.session_state.chatbot_waiting_response = False

if "chatbot_pending_user_message" not in st.session_state:
    st.session_state.chatbot_pending_user_message = None

if "tts_future" not in st.session_state:
    st.session_state.tts_future = None

if "tts_waiting_response" not in st.session_state:
    st.session_state.tts_waiting_response = False

if "active_tts_request_id" not in st.session_state:
    st.session_state.active_tts_request_id = None

if "last_tts_audio_event_id" not in st.session_state:
    st.session_state.last_tts_audio_event_id = None

if "last_autoplayed_tts_audio_event_id" not in st.session_state:
    st.session_state.last_autoplayed_tts_audio_event_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_tts_audio" not in st.session_state:
    st.session_state.last_tts_audio = None

if "last_processed_event_id" not in st.session_state:
    st.session_state.last_processed_event_id = None

if "last_scroll_action_id" not in st.session_state:
    st.session_state.last_scroll_action_id = 0

if "scroll_anchor_index" not in st.session_state:
    st.session_state.scroll_anchor_index = 0

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


SCROLL_ANCHORS = [
    "voice-scroll-top",
    "voice-scroll-content",
    "voice-scroll-middle",
    "voice-scroll-bottom",
]


# ==========================================================
# HELPERS
# ==========================================================

def normalize_question(text: str) -> str:
    """
    Normalizes user text for deterministic dashboard answers.
    """

    if not text:
        return ""

    text = text.lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character for character in text
        if unicodedata.category(character) != "Mn"
    )

    text = text.replace("'", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
            [
                sys.executable,
                "-u",
                str(LISTENER_SCRIPT),
            ],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        LISTENER_PID_FILE.write_text(
            str(process.pid),
            encoding="utf-8",
        )

        st.session_state.listener_process = process

        return True, f"Live listener démarré automatiquement. PID: {process.pid}"

    except Exception as error:
        return False, f"Impossible de démarrer le live listener : {error}"


def build_dashboard_context() -> str:
    """
    Builds a clear context for Ollama using only the data available
    in the Streamlit dashboard.
    """

    current_page = st.session_state.get("current_page", "resume")
    selected_metric = st.session_state.get("selected_metric")
    selected_dimension = st.session_state.get("selected_dimension")

    total_sales = int(sales_by_month["ventes"].sum())
    total_clients = int(sales_by_month["clients"].sum())
    average_sales = int(sales_by_month["ventes"].mean())

    best_month_row = sales_by_month.loc[sales_by_month["ventes"].idxmax()]
    worst_month_row = sales_by_month.loc[sales_by_month["ventes"].idxmin()]
    best_region_row = sales_by_region.loc[sales_by_region["ventes"].idxmax()]
    best_client_region_row = sales_by_region.loc[sales_by_region["clients"].idxmax()]

    sales_by_month_text = sales_by_month.to_dict(orient="records")
    sales_by_region_text = sales_by_region.to_dict(orient="records")

    context = f"""
Tu es l'assistant vocal d'un dashboard Streamlit local de démonstration.
Tu dois répondre uniquement à partir des données intégrées dans ce dashboard.
Tu ne dois jamais parler du site officiel Streamlit, du trafic web réel ou de données en temps réel.

Page actuelle du dashboard : {current_page}
Métrique sélectionnée : {selected_metric}
Dimension sélectionnée : {selected_dimension}

Données disponibles :
1. sales_by_month : ventes et clients par mois.
{sales_by_month_text}

2. sales_by_region : ventes et clients par région.
{sales_by_region_text}

Indicateurs déjà calculés :
- Ventes totales : {total_sales} €
- Clients totaux : {total_clients}
- Vente moyenne mensuelle : {average_sales} €
- Meilleur mois en ventes : {best_month_row["mois"]} avec {int(best_month_row["ventes"])} €
- Mois le plus faible en ventes : {worst_month_row["mois"]} avec {int(worst_month_row["ventes"])} €
- Meilleure région en ventes : {best_region_row["region"]} avec {int(best_region_row["ventes"])} €
- Région avec le plus de clients : {best_client_region_row["region"]} avec {int(best_client_region_row["clients"])} clients

Règles :
- Réponds en français.
- Réponds comme si tu analysais ce dashboard précis.
- Si l'utilisateur dit "visiteurs", comprends-le comme "clients" dans ce prototype.
- Si l'utilisateur demande une moyenne, un total, un maximum ou une comparaison, calcule avec les données fournies.
- Si une donnée n'existe vraiment pas, dis : "Cette donnée n'est pas présente dans le jeu de données actuel."
- Ne recommande pas de contacter Streamlit.
- Ne parle pas de données temps réel.
- Réponse courte, claire, utile et lisible à voix haute.
"""

    return context.strip()


def answer_direct_dashboard_question(user_message: str) -> str | None:
    """
    Answers common dashboard questions directly without calling Ollama.

    This improves speed and prevents wrong answers for obvious KPIs.
    """

    normalized_text = normalize_question(user_message)

    if not normalized_text:
        return None

    total_sales = int(sales_by_month["ventes"].sum())
    total_clients = int(sales_by_month["clients"].sum())
    average_sales = int(sales_by_month["ventes"].mean())

    best_month_row = sales_by_month.loc[sales_by_month["ventes"].idxmax()]
    worst_month_row = sales_by_month.loc[sales_by_month["ventes"].idxmin()]
    best_region_row = sales_by_region.loc[sales_by_region["ventes"].idxmax()]
    best_client_region_row = sales_by_region.loc[sales_by_region["clients"].idxmax()]

    asks_average_sales = (
        "vente moyenne" in normalized_text
        or "ventes moyenne" in normalized_text
        or "moyenne des ventes" in normalized_text
        or "moyenne de vente" in normalized_text
        or "moyenne vente" in normalized_text
    )

    if asks_average_sales:
        return (
            f"La vente moyenne mensuelle est de {average_sales:,} €. "
            "Elle est calculée à partir des ventes des six mois du dashboard."
        )

    asks_total_sales = (
        "vente totale" in normalized_text
        or "ventes totales" in normalized_text
        or "total des ventes" in normalized_text
        or "chiffre d affaire total" in normalized_text
        or "ca total" in normalized_text
    )

    if asks_total_sales:
        return f"Les ventes totales du dashboard sont de {total_sales:,} €."

    asks_total_clients = (
        "client total" in normalized_text
        or "clients total" in normalized_text
        or "clients totaux" in normalized_text
        or "total des clients" in normalized_text
        or "visiteurs" in normalized_text
    )

    if asks_total_clients:
        return (
            f"Le dashboard contient {total_clients} clients au total. "
            f"La région avec le plus de clients est {best_client_region_row['region']}, "
            f"avec {int(best_client_region_row['clients'])} clients."
        )

    asks_best_region = (
        "meilleure region" in normalized_text
        or "region vend le plus" in normalized_text
        or "region a le plus de ventes" in normalized_text
        or "quelle region vend" in normalized_text
    )

    if asks_best_region:
        return (
            f"La région qui vend le plus est {best_region_row['region']}, "
            f"avec {int(best_region_row['ventes']):,} € de ventes."
        )

    asks_best_month = (
        "meilleur mois" in normalized_text
        or "mois vend le plus" in normalized_text
        or "mois avec le plus de ventes" in normalized_text
    )

    if asks_best_month:
        return (
            f"Le meilleur mois est {best_month_row['mois']}, "
            f"avec {int(best_month_row['ventes']):,} € de ventes."
        )

    asks_worst_month = (
        "pire mois" in normalized_text
        or "mois le plus faible" in normalized_text
        or "mois vend le moins" in normalized_text
    )

    if asks_worst_month:
        return (
            f"Le mois le plus faible est {worst_month_row['mois']}, "
            f"avec {int(worst_month_row['ventes']):,} € de ventes."
        )

    return None


def call_chatbot_in_background(
    user_message: str,
    conversation_history: list[dict],
    dashboard_context: str,
) -> str:
    """
    Calls Ollama in a background thread.

    This function only generates the text response.
    TTS is handled separately so the UI can display text before audio.
    """

    if ask_ollama is None:
        return (
            "Ollama n'est pas encore connecté dans l'application. "
            "Vérifiez le fichier src/ollama_client.py."
        )

    try:
        assistant_response = ask_ollama(
            user_message=user_message,
            conversation_history=conversation_history,
            dashboard_context=dashboard_context,
        )

        return assistant_response

    except Exception:
        return (
            "Ollama n'est pas disponible pour le moment. "
            "Vérifiez que le modèle llama3.2:latest est installé."
        )


def call_tts_in_background(
    text: str,
    tts_request_id: str,
) -> tuple[str, str | None]:
    """
    Generates Kokoro TTS audio in a background thread.

    Returns the request id with the audio path so Streamlit can ignore
    old TTS results if a new question was already asked.
    """

    if synthesize_response_to_wav is None:
        return tts_request_id, None

    try:
        audio_path = synthesize_response_to_wav(text)
        return tts_request_id, audio_path

    except Exception:
        return tts_request_id, None


def start_tts_for_response(assistant_response: str):
    """
    Starts Kokoro TTS asynchronously for the latest assistant response.
    """

    if synthesize_response_to_wav is None:
        return

    tts_request_id = f"tts_{int(time.time() * 1000)}"

    st.session_state.active_tts_request_id = tts_request_id
    st.session_state.tts_waiting_response = True
    st.session_state.last_tts_audio = None
    st.session_state.last_tts_audio_event_id = None

    tts_executor = get_tts_executor()

    st.session_state.tts_future = tts_executor.submit(
        call_tts_in_background,
        assistant_response,
        tts_request_id,
    )


def run_scroll_if_needed():
    """
    Executes stable voice scroll using HTML anchors.
    """

    current_scroll_id = st.session_state.get("scroll_action_id", 0)
    last_scroll_id = st.session_state.get("last_scroll_action_id", 0)

    if current_scroll_id == last_scroll_id:
        return

    direction = st.session_state.get("pending_scroll_direction")
    amount = int(st.session_state.get("pending_scroll_amount", 700))

    if not direction:
        return

    current_anchor_index = int(st.session_state.get("scroll_anchor_index", 0))

    step = 1

    if amount >= 1200:
        step = 2

    if direction == "down":
        next_anchor_index = min(
            current_anchor_index + step,
            len(SCROLL_ANCHORS) - 1,
        )

    elif direction == "up":
        next_anchor_index = max(
            current_anchor_index - step,
            0,
        )

    elif direction == "top":
        next_anchor_index = 0

    elif direction == "bottom":
        next_anchor_index = len(SCROLL_ANCHORS) - 1

    else:
        return

    target_anchor = SCROLL_ANCHORS[next_anchor_index]

    st.session_state.scroll_anchor_index = next_anchor_index
    st.session_state.last_scroll_action_id = current_scroll_id

    safe_target_anchor = escape(target_anchor)

    components.html(
        f"""
        <script>
        function scrollToVoiceAnchor() {{
            try {{
                const targetHash = "#{safe_target_anchor}";
                const parentWindow = window.parent;

                if (parentWindow.location.hash === targetHash) {{
                    parentWindow.history.replaceState(
                        null,
                        "",
                        parentWindow.location.pathname + parentWindow.location.search
                    );
                }}

                setTimeout(function () {{
                    parentWindow.location.hash = targetHash;
                }}, 80);

            }} catch (error) {{
                window.location.hash = "#{safe_target_anchor}";
            }}
        }}

        scrollToVoiceAnchor();
        setTimeout(scrollToVoiceAnchor, 250);
        </script>
        """,
        height=0,
    )


def process_chatbot_message(user_message: str):
    """
    Sends the user message to Ollama or answers directly when possible.
    """

    if not user_message:
        return

    if st.session_state.chatbot_waiting_response:
        st.session_state.status_message = "Le chatbot traite déjà une réponse."
        return

    st.session_state.last_tts_audio = None
    st.session_state.last_tts_audio_event_id = None
    st.session_state.active_tts_request_id = None
    st.session_state.tts_waiting_response = False
    st.session_state.chatbot_open = True

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    direct_answer = answer_direct_dashboard_question(user_message)

    if direct_answer:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": direct_answer,
            }
        )

        st.session_state.status_message = "Réponse calculée depuis les données du dashboard."
        start_tts_for_response(direct_answer)
        return

    conversation_history = st.session_state.chat_history[-4:].copy()
    dashboard_context = build_dashboard_context()

    executor = get_chatbot_executor()

    st.session_state.chatbot_future = executor.submit(
        call_chatbot_in_background,
        user_message,
        conversation_history,
        dashboard_context,
    )

    st.session_state.chatbot_waiting_response = True
    st.session_state.chatbot_pending_user_message = user_message
    st.session_state.status_message = "Question envoyée au chatbot."


def check_chatbot_background_response():
    """
    Checks if Ollama has finished.

    When the text response is ready:
    - displays it immediately
    - starts Kokoro TTS in a separate background task
    """

    future = st.session_state.get("chatbot_future")

    if future is None:
        return

    if not future.done():
        return

    try:
        assistant_response = future.result()

    except Exception as error:
        assistant_response = f"Erreur chatbot : {error}"

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )

    st.session_state.chatbot_future = None
    st.session_state.chatbot_waiting_response = False
    st.session_state.chatbot_pending_user_message = None
    st.session_state.ollama_available = True
    st.session_state.status_message = "Réponse chatbot reçue."

    start_tts_for_response(assistant_response)


def check_tts_background_response():
    """
    Checks if Kokoro TTS has finished.

    Old TTS results are ignored if a newer question was asked.
    """

    future = st.session_state.get("tts_future")

    if future is None:
        return

    if not future.done():
        return

    try:
        tts_request_id, audio_path = future.result()

    except Exception:
        tts_request_id = None
        audio_path = None

    st.session_state.tts_future = None
    st.session_state.tts_waiting_response = False

    if tts_request_id != st.session_state.get("active_tts_request_id"):
        return

    st.session_state.last_tts_audio = audio_path

    if audio_path:
        st.session_state.last_tts_audio_event_id = tts_request_id
        st.session_state.status_message = "Audio Kokoro généré."
    else:
        st.session_state.last_tts_audio_event_id = None
        st.session_state.status_message = "Réponse texte reçue, mais audio Kokoro indisponible."


def process_voice_command(command: dict):
    """
    Routes a parsed voice command.
    """

    intent = command.get("intent")

    if intent == "open_chatbot":
        st.session_state.chatbot_open = True
        st.session_state.status_message = "Chatbot ouvert. Posez votre question."
        return

    if intent == "close_chatbot":
        st.session_state.chatbot_open = False
        st.session_state.last_tts_audio = None
        st.session_state.last_tts_audio_event_id = None
        st.session_state.last_autoplayed_tts_audio_event_id = None
        st.session_state.active_tts_request_id = None
        st.session_state.tts_waiting_response = False
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


def get_audio_html(
    audio_path: str | None,
    autoplay: bool = False,
    audio_event_id: str | None = None,
) -> str:
    """
    Creates invisible autoplay audio.

    The audio is embedded inside the chatbot HTML only when a new
    TTS event must be played. No visible controls are displayed.
    """

    if not audio_path:
        return ""

    path = Path(audio_path)

    if not path.exists():
        return ""

    if not autoplay:
        return ""

    try:
        audio_bytes = path.read_bytes()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        if not audio_event_id:
            audio_event_id = f"kokoro_audio_{int(time.time() * 1000)}"

        safe_audio_id = escape(audio_event_id)

        return f"""
        <audio
            id="{safe_audio_id}"
            autoplay
            preload="auto"
            style="display: none; width: 0; height: 0; opacity: 0; pointer-events: none;"
        >
            <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
        </audio>

        <script>
        setTimeout(function () {{
            const audio = document.getElementById("{safe_audio_id}");
            if (audio) {{
                audio.currentTime = 0;
                audio.play().catch(function(error) {{
                    console.log("Autoplay blocked:", error);
                }});
            }}
        }}, 200);
        </script>
        """

    except Exception:
        return ""


def render_html(html_content: str):
    """
    Renders HTML safely in Streamlit.
    """

    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def render_scroll_anchor(anchor_id: str):
    """
    Renders an invisible anchor used for stable voice scrolling.
    """

    safe_anchor_id = escape(anchor_id)

    render_html(
        f"""
        <div id="{safe_anchor_id}" style="height: 1px; width: 1px;"></div>
        """
    )


def render_listener_indicator():
    """
    Displays a red/green listener indicator and the current listener text.
    """

    listener_data = read_listener_status()

    is_active = False
    message = "Live listener non démarré."

    if listener_data:
        updated_at = listener_data.get("updated_at", 0)
        age = time.time() - updated_at

        if age < 10:
            is_active = bool(listener_data.get("is_active", False))
            message = listener_data.get("message") or "Listener actif."
        else:
            is_active = False
            message = "Aucun signal récent du listener."

    dot_color = "#22c55e" if is_active else "#ef4444"
    shadow_color = "rgba(34, 197, 94, 0.18)" if is_active else "rgba(239, 68, 68, 0.18)"
    status_label = "Écoute active" if is_active else "Écoute inactive"

    safe_message = escape(message)
    safe_status_label = escape(status_label)

    html = f"""
    <style>
    .listener-status-wrapper {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: -4px;
        margin-bottom: 16px;
        padding: 10px 14px;
        border-radius: 14px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        color: #111827;
        font-size: 14px;
    }}

    .listener-dot {{
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: {dot_color};
        box-shadow: 0 0 0 4px {shadow_color};
        flex-shrink: 0;
    }}

    .listener-text {{
        display: flex;
        flex-direction: column;
        line-height: 1.3;
    }}

    .listener-title {{
        font-weight: 700;
        font-size: 13px;
    }}

    .listener-message {{
        font-size: 13px;
        color: #4b5563;
    }}
    </style>

    <div class="listener-status-wrapper">
        <div class="listener-dot"></div>
        <div class="listener-text">
            <div class="listener-title">{safe_status_label}</div>
            <div class="listener-message">{safe_message}</div>
        </div>
    </div>
    """

    render_html(html)


def render_chatbot_popup():
    """
    Renders the floating chatbot button or a stable chat popup.
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
        z-index: 999999;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
        opacity: 1 !important;
        filter: none !important;
    }

    .chatbot-popup {
        position: fixed;
        right: 24px;
        bottom: 100px;
        width: 390px;
        height: 540px;
        background: #f3f4f6 !important;
        color: #111827;
        border-radius: 24px;
        z-index: 999999;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.24);
        border: 1px solid #e5e7eb;
        font-family: Arial, sans-serif;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        opacity: 1 !important;
        filter: none !important;
        backdrop-filter: none !important;
    }

    .chatbot-header {
        background: #111827;
        color: white;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        opacity: 1 !important;
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
        background: #f9fafb !important;
        opacity: 1 !important;
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
        opacity: 1 !important;
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

    .bubble-thinking {
        background: white;
        color: #6b7280;
        border-bottom-left-radius: 5px;
        border: 1px solid #e5e7eb;
        font-style: italic;
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
        for message in st.session_state.chat_history[-12:]:
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

    if st.session_state.get("chatbot_waiting_response"):
        messages_html += """
        <div class="message-row assistant-row">
            <div class="bubble bubble-thinking">
                <span class="bubble-label">Assistant</span>
                Réflexion en cours...
            </div>
        </div>
        """

    if st.session_state.get("tts_waiting_response"):
        messages_html += """
        <div class="message-row assistant-row">
            <div class="bubble bubble-thinking">
                <span class="bubble-label">Audio</span>
                Préparation de la réponse vocale...
            </div>
        </div>
        """

    audio_event_id = st.session_state.get("last_tts_audio_event_id")
    last_autoplayed_id = st.session_state.get("last_autoplayed_tts_audio_event_id")

    should_autoplay = (
        st.session_state.last_tts_audio
        and audio_event_id
        and audio_event_id != last_autoplayed_id
    )

    audio_html = get_audio_html(
        st.session_state.last_tts_audio,
        autoplay=bool(should_autoplay),
        audio_event_id=audio_event_id,
    )

    if should_autoplay:
        st.session_state.last_autoplayed_tts_audio_event_id = audio_event_id

    ollama_status = (
        "Ollama actif"
        if st.session_state.get("ollama_available")
        else "Ollama en démarrage ou indisponible"
    )

    chatbot_scroll_script = """
    <script>
    function scrollChatbotToBottom() {
        const container = document.getElementById("chatbot-messages-container");
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    requestAnimationFrame(scrollChatbotToBottom);
    setTimeout(scrollChatbotToBottom, 150);
    setTimeout(scrollChatbotToBottom, 400);
    </script>
    """

    chatbot_html = f"""
    <div class="chatbot-popup">
        <div class="chatbot-header">
            <div class="chatbot-avatar">🤖</div>
            <div class="chatbot-title-block">
                <div class="chatbot-title">Assistant vocal</div>
                <div class="chatbot-status">{ollama_status}</div>
            </div>
        </div>

        <div class="chatbot-messages" id="chatbot-messages-container">
            {messages_html}
            {audio_html}
        </div>

        <div class="chatbot-footer">
            Dites <strong>chatbot désactivé</strong> ou <strong>ferme chatbot</strong> pour fermer la fenêtre.
        </div>
    </div>

    {chatbot_scroll_script}
    """

    render_html(chatbot_html)

# ==========================================================
# START LIVE LISTENER WHEN DASHBOARD STARTS
# ==========================================================

if not st.session_state.listener_checked:
    listener_available, listener_status = start_live_listener_if_needed()

    st.session_state.listener_available = listener_available
    st.session_state.listener_status = listener_status
    st.session_state.listener_checked = True


# ==========================================================
# PROCESS VOICE COMMANDS
# ==========================================================

read_and_process_latest_voice_event()
check_chatbot_background_response()
check_tts_background_response()


# ==========================================================
# UI
# ==========================================================

render_scroll_anchor("voice-scroll-top")

st.title("🎙️ Voice-Controlled Dashboard")

st.caption(
    "Dashboard contrôlé à la voix avec wake word, commandes vocales, scroll, "
    "mode chatbot local via Ollama et réponse vocale via Kokoro."
)

st.info(st.session_state.status_message)
render_listener_indicator()


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

render_scroll_anchor("voice-scroll-content")


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

    render_scroll_anchor("voice-scroll-middle")

    st.markdown("### Ventes par région")
    st.bar_chart(
        sales_by_region.set_index("region")["ventes"]
    )


elif current_page == "ventes":
    st.markdown("### Analyse des ventes")

    st.line_chart(
        sales_by_month.set_index("mois")["ventes"]
    )

    render_scroll_anchor("voice-scroll-middle")

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

    render_scroll_anchor("voice-scroll-middle")

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

    render_scroll_anchor("voice-scroll-middle")

    st.dataframe(sales_by_region, use_container_width=True)


render_scroll_anchor("voice-scroll-bottom")


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

    st.write("Chatbot attend une réponse :")
    st.write(st.session_state.chatbot_waiting_response)

    st.write("TTS attend une réponse :")
    st.write(st.session_state.tts_waiting_response)

    st.write("Dernier audio TTS :")
    st.code(st.session_state.get("last_tts_audio"))

    st.write("Kokoro TTS disponible :")
    st.write(synthesize_response_to_wav is not None)

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
# VOICE SCROLL
# ==========================================================

run_scroll_if_needed()


# ==========================================================
# AUTO REFRESH
# ==========================================================

st_autorefresh(
    interval=1000,
    key="voice_dashboard_autorefresh",
)