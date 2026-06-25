import os
import queue
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from stt_engine import LocalSTTEngine
from wake_word import WakeWordDetector
from command_parser import CommandParser
from command_bus import write_command_event, write_listener_status


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
TEMP_AUDIO_FILE = RUNTIME_DIR / "listener_phrase.wav"


# ==========================================================
# FIXED MICROPHONE CONFIGURATION
# ==========================================================

FIXED_INPUT_DEVICE = 1
FIXED_SAMPLE_RATE = 44100
FIXED_SPEECH_THRESHOLD = 0.0065


# ==========================================================
# RECORDING CONFIGURATION
# ==========================================================

FRAME_DURATION_SECONDS = 0.25
START_SPEECH_FRAMES = 3
END_SILENCE_SECONDS = 0.9
MIN_SPEECH_SECONDS = 0.7
MAX_UTTERANCE_SECONDS = 8
PRE_SPEECH_SECONDS = 0.75
PRE_SPEECH_FRAMES = int(PRE_SPEECH_SECONDS / FRAME_DURATION_SECONDS)

FOLLOW_UP_SECONDS = 3
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")


def print_terminal(label: str, message: str = ""):
    if message:
        print(f"[{label}] {message}", flush=True)
    else:
        print(f"[{label}]", flush=True)

def publish_listener_status(
    status: str,
    message: str,
    is_active: bool = True,
    transcription: str | None = None,
):
    """
    Publishes listener status to Streamlit.
    """

    try:
        write_listener_status(
            status=status,
            message=message,
            is_active=is_active,
            transcription=transcription,
        )
    except Exception:
        pass

class PersistentMicrophone:
    """
    Keeps the microphone stream open permanently.

    This avoids the Windows/Realtek issue where the mic only works properly
    when another app, like a screen recorder, keeps it active.
    """

    def __init__(self, input_device: int, sample_rate: int):
        self.input_device = input_device
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print_terminal("AUDIO WARNING", str(status))

        audio_float = indata.astype(np.float32) / 32768.0
        self.audio_queue.put(audio_float.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.input_device,
            channels=1,
            dtype="int16",
            blocksize=int(FRAME_DURATION_SECONDS * self.sample_rate),
            callback=self._callback,
        )

        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def read_frame(self) -> tuple[np.ndarray, float]:
        frame = self.audio_queue.get()
        energy = float(np.sqrt(np.mean(frame ** 2)))
        return frame, energy


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))

    if peak < 0.0001:
        return audio

    target_peak = 0.85
    gain = target_peak / peak
    gain = min(gain, 12.0)

    return audio * gain


def record_full_phrase(
    microphone: PersistentMicrophone,
    sample_rate: int,
    speech_threshold: float
) -> tuple[str | None, float, float]:
    """
    Waits until speech starts, then records until silence is detected.

    Uses a pre-speech buffer to avoid cutting the beginning of the sentence.
    This improves wake words like "Ok Jack".
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    publish_listener_status(
        status="listening",
        message="En écoute... dites Jack ou Ok Jack.",
        is_active=True,
    )

    microphone.clear_queue()

    frames = []
    pre_speech_buffer = deque(maxlen=PRE_SPEECH_FRAMES)

    is_recording = False
    silence_duration = 0.0
    speech_duration = 0.0
    max_energy = 0.0
    start_time = None
    above_threshold_frames = 0

    while True:
        frame, energy = microphone.read_frame()
        max_energy = max(max_energy, energy)

        if not is_recording:
            pre_speech_buffer.append(frame)

            print(
                f"\r[WAITING] niveau micro {energy:.6f} | seuil {speech_threshold:.6f}",
                end="",
                flush=True,
            )

            if energy >= speech_threshold:
                above_threshold_frames += 1
            else:
                above_threshold_frames = 0

            if above_threshold_frames >= START_SPEECH_FRAMES:
                is_recording = True
                start_time = time.time()

                frames.extend(list(pre_speech_buffer))
                speech_duration += FRAME_DURATION_SECONDS * len(pre_speech_buffer)
                silence_duration = 0.0

                print()
                print_terminal(
                    "RECORDING",
                    "Voix détectée, enregistrement de la phrase avec pre-roll..."
                )

                publish_listener_status(
                    status="recording",
                    message="Enregistrement en cours...",
                    is_active=True,
                )

            continue

        frames.append(frame)
        speech_duration += FRAME_DURATION_SECONDS

        if energy < speech_threshold:
            silence_duration += FRAME_DURATION_SECONDS
        else:
            silence_duration = 0.0

        elapsed = time.time() - start_time if start_time else speech_duration

        print(
            f"\r[RECORDING] {elapsed:.1f}s | niveau {energy:.6f} | silence {silence_duration:.1f}s",
            end="",
            flush=True,
        )

        if silence_duration >= END_SILENCE_SECONDS:
            print()
            break

        if elapsed >= MAX_UTTERANCE_SECONDS:
            print()
            print_terminal("RECORDING", "Durée maximum atteinte, transcription...")
            break

    if speech_duration < MIN_SPEECH_SECONDS:
        print_terminal("IGNORED", "Audio trop court.")

        publish_listener_status(
            status="listening",
            message="Audio trop court ignoré. En écoute...",
            is_active=True,
        )

        return None, speech_duration, max_energy

    audio = np.concatenate(frames, axis=0)
    audio = normalize_audio(audio)

    sf.write(TEMP_AUDIO_FILE, audio, sample_rate)

    return str(TEMP_AUDIO_FILE), speech_duration, max_energy


def send_command_to_dashboard(
    transcription: str,
    wake_result: dict,
    parsed_command: dict
):
    write_command_event(
        transcription=transcription,
        wake_result=wake_result,
        parsed_command=parsed_command,
    )

    print_terminal("COMMAND", str(parsed_command))
    print()


def process_transcription(
    transcription: str,
    wake_detector: WakeWordDetector,
    parser: CommandParser,
    waiting_for_follow_up_command: bool,
    chatbot_mode: bool,
) -> tuple[bool, bool]:
    """
    Processes a transcription.

    Important behavior:
    - Normal commands require the wake word.
    - "Ok Jack, chatbot" opens temporary chatbot mode.
    - In chatbot mode, the next phrase is sent to Ollama.
    - After one chatbot message, chatbot mode stops automatically.
    - "chatbot désactivé" or "ferme chatbot" closes the chatbot.
    """

    transcription = transcription.strip()

    if not transcription:
        print_terminal("TEXT", "Transcription vide.")
        print()
        return waiting_for_follow_up_command, chatbot_mode

    print_terminal("TEXT", transcription)

    wake_result = wake_detector.extract_command(transcription)

    # ==========================================================
    # CASE 1: WAKE WORD DETECTED
    # ==========================================================

    if wake_result["activated"]:
        command_text = wake_result["command_text"]

        print_terminal("WAKE", "Ok Jack détecté.")

        if not command_text:
            print_terminal(
                "SESSION",
                f"Ok Jack détecté. En attente de la commande pendant {FOLLOW_UP_SECONDS}s."
            )
            print()
            return True, chatbot_mode

        parsed_command = parser.parse(command_text)
        intent = parsed_command.get("intent")

        if intent == "open_chatbot":
            chatbot_mode = True
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Mode chatbot activé pour la prochaine phrase.")
            return False, chatbot_mode

        if intent == "chatbot_message":
            # Direct command: "Ok Jack, chatbot explique les ventes"
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Message envoyé au chatbot.")
            print_terminal("CHATBOT", "Mode chatbot arrêté après la commande.")
            return False, False

        if intent == "close_chatbot":
            chatbot_mode = False
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Chatbot fermé.")
            return False, chatbot_mode

        send_command_to_dashboard(transcription, wake_result, parsed_command)
        return False, chatbot_mode

    # ==========================================================
    # CASE 2: CHATBOT MODE ACTIVE
    # ==========================================================

    if chatbot_mode:
        parsed_close_command = parser.parse(transcription)

        if parsed_close_command.get("intent") == "close_chatbot":
            wake_result = {
                "activated": True,
                "wake_word": "chatbot_mode",
                "command_text": transcription,
                "original_text": transcription,
                "mode": "chatbot_close"
            }

            send_command_to_dashboard(transcription, wake_result, parsed_close_command)
            print_terminal("CHATBOT", "Chatbot fermé.")
            return False, False

        parsed_command = {
            "intent": "chatbot_message",
            "message": transcription,
            "raw_text": transcription
        }

        wake_result = {
            "activated": True,
            "wake_word": "chatbot_mode",
            "command_text": transcription,
            "original_text": transcription,
            "mode": "chatbot_message"
        }

        send_command_to_dashboard(transcription, wake_result, parsed_command)

        print_terminal("CHATBOT", "Message envoyé à Ollama.")
        print_terminal("CHATBOT", "Mode chatbot arrêté automatiquement après la réponse.")

        # IMPORTANT:
        # The chatbot mode stops after one message.
        return False, False

    # ==========================================================
    # CASE 3: FOLLOW-UP AFTER ONLY "OK JACK"
    # ==========================================================

    if waiting_for_follow_up_command:
        parsed_command = parser.parse(transcription)
        intent = parsed_command.get("intent")

        if intent == "unknown":
            print_terminal("FOLLOW-UP", transcription)
            print_terminal("ERROR", "Commande non reconnue, réessayez.")
            print()
            return True, chatbot_mode

        wake_result = {
            "activated": True,
            "wake_word": "ok jack",
            "command_text": transcription,
            "original_text": transcription,
            "mode": "follow_up_command",
        }

        if intent == "open_chatbot":
            chatbot_mode = True
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Mode chatbot activé pour la prochaine phrase.")
            return False, chatbot_mode

        if intent == "chatbot_message":
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Message envoyé au chatbot.")
            print_terminal("CHATBOT", "Mode chatbot arrêté après la commande.")
            return False, False

        if intent == "close_chatbot":
            chatbot_mode = False
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Chatbot fermé.")
            return False, chatbot_mode

        send_command_to_dashboard(transcription, wake_result, parsed_command)
        return False, chatbot_mode

    # ==========================================================
    # CASE 4: NO WAKE WORD AND NO CHATBOT MODE
    # ==========================================================

    print_terminal("IGNORED", "Pas de wake word, phrase ignorée.")
    print()

    return waiting_for_follow_up_command, chatbot_mode


def main():
    input_device = FIXED_INPUT_DEVICE
    sample_rate = FIXED_SAMPLE_RATE
    speech_threshold = FIXED_SPEECH_THRESHOLD

    try:
        device_name = sd.query_devices(input_device)["name"]
        print_terminal(
            "MIC",
            f"Using persistent microphone: {device_name} | device {input_device} | {sample_rate} Hz"
        )
    except Exception:
        print_terminal(
            "MIC",
            f"Using persistent microphone device {input_device} | {sample_rate} Hz"
        )

    publish_listener_status(
        status="starting",
        message="Chargement du modèle Whisper...",
        is_active=False,
    )

    print_terminal("START", f"Chargement de Whisper model: {MODEL_SIZE}")
    stt = LocalSTTEngine(model_size=MODEL_SIZE)

    wake_detector = WakeWordDetector()
    parser = CommandParser()

    waiting_for_follow_up_command = False
    follow_up_deadline = 0.0

    chatbot_mode = False

    microphone = PersistentMicrophone(
        input_device=input_device,
        sample_rate=sample_rate
    )

    print()
    print_terminal("LISTENER", "Phrase-based persistent microphone listener started.")
    print_terminal("INFO", "Le micro reste ouvert en continu pour éviter les problèmes Realtek/Windows.")
    print_terminal("CONFIG", f"Input device: {input_device}")
    print_terminal("CONFIG", f"Sample rate: {sample_rate}")
    print_terminal("CONFIG", f"Speech threshold: {speech_threshold}")
    print_terminal("CONFIG", f"End silence: {END_SILENCE_SECONDS}s")
    print_terminal("EXAMPLE", "Say: Ok Jack, chatbot")
    print_terminal("EXAMPLE", "Then say: Explique les ventes par région")
    print_terminal("STOP", "Press CTRL + C to stop.")
    print()

    try:
        microphone.start()
        print_terminal("MIC", "Microphone stream opened and kept alive.")
        publish_listener_status(
            status="listening",
            message="En écoute... dites Jack, puis une commande.",
            is_active=True,
        )
        print()

        while True:
            if waiting_for_follow_up_command and time.time() > follow_up_deadline:
                waiting_for_follow_up_command = False
                print_terminal("SESSION", "Fermée après délai.")
                print()

            audio_path, duration, max_energy = record_full_phrase(
                microphone=microphone,
                sample_rate=sample_rate,
                speech_threshold=speech_threshold
            )

            if audio_path is None:
                continue

            print_terminal(
                "TRANSCRIBING",
                f"Phrase capturée ({duration:.1f}s, niveau max {max_energy:.6f})..."
            )

            publish_listener_status(
                status="transcribing",
                message=f"Transcription en cours... audio capturé : {duration:.1f}s",
                is_active=True,
            )

            transcription = stt.transcribe_audio(
                audio_path,
                language="fr"
            ).strip()

            publish_listener_status(
                status="transcribed",
                message=f"Texte entendu : {transcription}",
                is_active=True,
                transcription=transcription,
            )

            now = time.time()

            waiting_for_follow_up_command, chatbot_mode = process_transcription(
                transcription=transcription,
                wake_detector=wake_detector,
                parser=parser,
                waiting_for_follow_up_command=waiting_for_follow_up_command,
                chatbot_mode=chatbot_mode,
            )

            if waiting_for_follow_up_command:
                follow_up_deadline = now + FOLLOW_UP_SECONDS

            if chatbot_mode:
                print_terminal("MODE", "Chatbot actif. Les prochaines phrases iront à Ollama.")
            else:
                print_terminal("MODE", "Mode commandes dashboard.")
            print()

    except KeyboardInterrupt:
        print()
        print_terminal("STOP", "Phrase listener stopped.")

    except Exception as error:
        print()
        print_terminal("ERROR", str(error))

        publish_listener_status(
            status="error",
            message=f"Erreur listener : {error}",
            is_active=False,
        )

    finally:
        microphone.stop()
        publish_listener_status(
            status="stopped",
            message="Live listener arrêté.",
            is_active=False,
        )
        print_terminal("MIC", "Microphone stream closed.")


if __name__ == "__main__":
    main()