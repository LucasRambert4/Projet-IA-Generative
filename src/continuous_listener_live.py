import os
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from stt_engine import LocalSTTEngine
from wake_word import WakeWordDetector
from command_parser import CommandParser

from command_bus import (
    write_command_event,
    write_listener_status,
    read_listener_control,
)
from vad_engine import SileroVADEngine, RollingAudioBuffer


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
TEMP_AUDIO_FILE = RUNTIME_DIR / "listener_phrase.wav"


# ==========================================================
# MICROPHONE CONFIGURATION
# ==========================================================

FIXED_INPUT_DEVICE = 1
FIXED_SAMPLE_RATE = 44100


# ==========================================================
# VAD / RECORDING CONFIGURATION
# ==========================================================

FRAME_DURATION_SECONDS = 0.10

# VAD checks a rolling window, not only the last audio frame.
VAD_WINDOW_SECONDS = 0.55
VAD_WINDOW_FRAMES = int(VAD_WINDOW_SECONDS / FRAME_DURATION_SECONDS)

# Pre-roll keeps audio before detection so "Ok Jack" is not cut.
PRE_SPEECH_SECONDS = 2.0
PRE_SPEECH_FRAMES = int(PRE_SPEECH_SECONDS / FRAME_DURATION_SECONDS)

# Start recording only after several VAD-positive frames.
START_SPEECH_VAD_FRAMES = 2

# Stop only after real silence, not after a tiny pause.
END_SILENCE_SECONDS = 1.10

MIN_SPEECH_SECONDS = 0.65
MAX_UTTERANCE_SECONDS = 9.0

FOLLOW_UP_SECONDS = 4.0

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")


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


def get_listener_pause_remaining() -> tuple[float, str]:
    """
    Returns how many seconds the listener should remain paused.

    The pause is controlled by dashboard/app.py through runtime/listener_control.json.
    """

    control = read_listener_control()

    if not control:
        return 0.0, ""

    try:
        pause_until = float(control.get("pause_until", 0.0))
    except Exception:
        return 0.0, ""

    remaining = pause_until - time.time()

    if remaining <= 0:
        return 0.0, ""

    reason = control.get("reason", "pause_requested")

    return remaining, reason


def wait_if_listener_paused(microphone) -> bool:
    """
    Pauses the microphone listener when the assistant is speaking.

    Returns True if the caller should skip the current loop.
    """

    remaining, reason = get_listener_pause_remaining()

    if remaining <= 0:
        return False

    publish_listener_status(
        status="paused",
        message=f"Micro en pause : {reason}. Reprise dans {remaining:.1f}s.",
        is_active=False,
    )

    print_terminal(
        "PAUSED",
        f"Listener paused for {remaining:.1f}s | reason={reason}",
    )

    microphone.clear_queue()

    time.sleep(min(0.5, remaining))

    microphone.clear_queue()

    return True

class PersistentMicrophone:
    """
    Keeps the microphone stream open permanently.

    This avoids Windows / Realtek delays caused by repeatedly opening
    and closing the microphone.
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
    """
    Normalizes captured audio before sending it to Whisper.
    """

    peak = float(np.max(np.abs(audio)))

    if peak < 0.0001:
        return audio

    target_peak = 0.85
    gain = target_peak / peak
    gain = min(gain, 8.0)

    return audio * gain


def record_full_phrase(
    microphone: PersistentMicrophone,
    sample_rate: int,
    vad_engine: SileroVADEngine,
) -> tuple[str | None, float, float]:
    """
    Waits for real speech using Silero VAD, then records until silence.

    Main improvement:
    - No raw threshold trigger.
    - Uses real voice detection.
    - Uses a 2 second pre-roll to avoid losing "Ok Jack".
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    microphone.clear_queue()

    recorded_frames = []
    pre_speech_buffer = RollingAudioBuffer(max_frames=PRE_SPEECH_FRAMES)
    vad_buffer = RollingAudioBuffer(max_frames=VAD_WINDOW_FRAMES)

    is_recording = False
    vad_positive_frames = 0
    silence_duration = 0.0
    speech_duration = 0.0
    max_energy = 0.0
    start_time = None
    last_status_update = 0.0

    while True:
        if wait_if_listener_paused(microphone):
            return None, 0.0, max_energy

        frame, frame_energy = microphone.read_frame()
        max_energy = max(max_energy, frame_energy)

        pre_speech_buffer.append(frame)
        vad_buffer.append(frame)

        vad_audio = vad_buffer.to_audio()
        speech_detected, vad_energy = vad_engine.contains_speech(
            audio=vad_audio,
            original_sample_rate=sample_rate,
        )

        if not is_recording:
            now = time.time()

            if now - last_status_update > 2.0:
                publish_listener_status(
                    status="listening",
                    message="En écoute VAD... dites Jack ou Ok Jack.",
                    is_active=True,
                )
                last_status_update = now

            if speech_detected:
                vad_positive_frames += 1
            else:
                vad_positive_frames = 0

            print(
                (
                    f"\r[WAITING VAD] speech={speech_detected} | "
                    f"vad_frames={vad_positive_frames}/{START_SPEECH_VAD_FRAMES} | "
                    f"energy={vad_energy:.6f}"
                ),
                end="",
                flush=True,
            )

            if vad_positive_frames >= START_SPEECH_VAD_FRAMES:
                is_recording = True
                start_time = time.time()
                silence_duration = 0.0

                recorded_frames.extend(pre_speech_buffer.to_list())
                speech_duration += FRAME_DURATION_SECONDS * len(pre_speech_buffer.to_list())

                print()
                print_terminal(
                    "RECORDING",
                    "Voix détectée par Silero VAD, enregistrement avec pre-roll..."
                )

                publish_listener_status(
                    status="recording",
                    message="Voix détectée. Enregistrement en cours...",
                    is_active=True,
                )

            continue

        recorded_frames.append(frame)
        speech_duration += FRAME_DURATION_SECONDS

        if speech_detected:
            silence_duration = 0.0
        else:
            silence_duration += FRAME_DURATION_SECONDS

        elapsed = time.time() - start_time if start_time else speech_duration

        print(
            (
                f"\r[RECORDING VAD] {elapsed:.1f}s | "
                f"speech={speech_detected} | "
                f"silence={silence_duration:.1f}s | "
                f"energy={vad_energy:.6f}"
            ),
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

    if not recorded_frames:
        return None, speech_duration, max_energy

    audio = np.concatenate(recorded_frames, axis=0)
    audio = normalize_audio(audio)

    sf.write(TEMP_AUDIO_FILE, audio, sample_rate)

    return str(TEMP_AUDIO_FILE), speech_duration, max_energy


def send_command_to_dashboard(
    transcription: str,
    wake_result: dict,
    parsed_command: dict,
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
    """

    transcription = transcription.strip()

    if not transcription:
        print_terminal("TEXT", "Transcription vide.")
        print()
        return waiting_for_follow_up_command, chatbot_mode

    print_terminal("TEXT", transcription)

    wake_result = wake_detector.extract_command(transcription)

    if wake_result["activated"]:
        command_text = wake_result["command_text"]

        print_terminal("WAKE", "Wake word détecté.")

        if not command_text:
            print_terminal(
                "SESSION",
                f"Wake word détecté. En attente de la commande pendant {FOLLOW_UP_SECONDS}s.",
            )
            print()
            return True, chatbot_mode

        parsed_command = parser.parse(command_text)
        intent = parsed_command.get("intent")

        if intent == "open_chatbot":
            chatbot_mode = True
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Mode chatbot activé pour les prochaines phrases.")
            return False, chatbot_mode

        if intent == "chatbot_message":
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Message envoyé au chatbot.")
            print_terminal("CHATBOT", "Mode chatbot conservé.")
            return False, True

        if intent == "close_chatbot":
            chatbot_mode = False
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Chatbot fermé.")
            return False, chatbot_mode

        send_command_to_dashboard(transcription, wake_result, parsed_command)
        return False, chatbot_mode

    if chatbot_mode:
        parsed_close_command = parser.parse(transcription)

        if parsed_close_command.get("intent") == "close_chatbot":
            wake_result = {
                "activated": True,
                "wake_word": "chatbot_mode",
                "command_text": transcription,
                "original_text": transcription,
                "mode": "chatbot_close",
            }

            send_command_to_dashboard(transcription, wake_result, parsed_close_command)
            print_terminal("CHATBOT", "Chatbot fermé.")
            return False, False

        parsed_command = {
            "intent": "chatbot_message",
            "message": transcription,
            "raw_text": transcription,
        }

        wake_result = {
            "activated": True,
            "wake_word": "chatbot_mode",
            "command_text": transcription,
            "original_text": transcription,
            "mode": "chatbot_message",
        }

        send_command_to_dashboard(transcription, wake_result, parsed_command)

        print_terminal("CHATBOT", "Message envoyé à Ollama.")
        print_terminal("CHATBOT", "Mode chatbot conservé pour la prochaine question.")

        return False, True

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
            "wake_word": "follow_up",
            "command_text": transcription,
            "original_text": transcription,
            "mode": "follow_up_command",
        }

        if intent == "open_chatbot":
            chatbot_mode = True
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Mode chatbot activé pour les prochaines phrases.")
            return False, chatbot_mode

        if intent == "chatbot_message":
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Message envoyé au chatbot.")
            print_terminal("CHATBOT", "Mode chatbot conservé.")
            return False, True

        if intent == "close_chatbot":
            chatbot_mode = False
            send_command_to_dashboard(transcription, wake_result, parsed_command)
            print_terminal("CHATBOT", "Chatbot fermé.")
            return False, chatbot_mode

        send_command_to_dashboard(transcription, wake_result, parsed_command)
        return False, chatbot_mode

    print_terminal("IGNORED", "Pas de wake word, phrase ignorée.")
    print()

    return waiting_for_follow_up_command, chatbot_mode


def main():
    input_device = FIXED_INPUT_DEVICE
    sample_rate = FIXED_SAMPLE_RATE

    try:
        device_name = sd.query_devices(input_device)["name"]
        print_terminal(
            "MIC",
            f"Using persistent microphone: {device_name} | device {input_device} | {sample_rate} Hz",
        )
    except Exception:
        print_terminal(
            "MIC",
            f"Using persistent microphone device {input_device} | {sample_rate} Hz",
        )

    publish_listener_status(
        status="starting",
        message="Chargement de Whisper et Silero VAD...",
        is_active=False,
    )

    print_terminal("START", f"Chargement de Whisper model: {MODEL_SIZE}")
    stt = LocalSTTEngine(model_size=MODEL_SIZE)

    print_terminal("START", "Chargement de Silero VAD...")
    vad_engine = SileroVADEngine(
        target_sample_rate=16000,
        speech_threshold=0.45,
        min_speech_duration_ms=80,
        min_silence_duration_ms=120,
    )

    wake_detector = WakeWordDetector()
    parser = CommandParser()

    waiting_for_follow_up_command = False
    follow_up_deadline = 0.0
    chatbot_mode = False

    microphone = PersistentMicrophone(
        input_device=input_device,
        sample_rate=sample_rate,
    )

    print()
    print_terminal("LISTENER", "Silero VAD listener started.")
    print_terminal("CONFIG", f"Input device: {input_device}")
    print_terminal("CONFIG", f"Sample rate: {sample_rate}")
    print_terminal("CONFIG", f"Frame duration: {FRAME_DURATION_SECONDS}s")
    print_terminal("CONFIG", f"VAD window: {VAD_WINDOW_SECONDS}s")
    print_terminal("CONFIG", f"Pre speech: {PRE_SPEECH_SECONDS}s")
    print_terminal("CONFIG", f"End silence: {END_SILENCE_SECONDS}s")
    print_terminal("EXAMPLE", "Say: Ok Jack chatbot")
    print_terminal("EXAMPLE", "Then say: Quelle est ma vente moyenne ?")
    print_terminal("STOP", "Press CTRL + C to stop.")
    print()

    try:
        microphone.start()
        print_terminal("MIC", "Microphone stream opened and kept alive.")

        publish_listener_status(
            status="listening",
            message="En écoute VAD... dites Jack ou Ok Jack.",
            is_active=True,
        )

        while True:
            if waiting_for_follow_up_command and time.time() > follow_up_deadline:
                waiting_for_follow_up_command = False
                print_terminal("SESSION", "Fermée après délai.")
                print()

            if wait_if_listener_paused(microphone):
                continue

            audio_path, duration, max_energy = record_full_phrase(
                microphone=microphone,
                sample_rate=sample_rate,
                vad_engine=vad_engine,
            )

            if audio_path is None:
                continue

            print_terminal(
                "TRANSCRIBING",
                f"Phrase capturée ({duration:.1f}s, niveau max {max_energy:.6f})...",
            )

            publish_listener_status(
                status="transcribing",
                message=f"Transcription en cours... audio capturé : {duration:.1f}s",
                is_active=True,
            )

            transcription = stt.transcribe_audio(
                audio_path,
                language="fr",
            ).strip()

            publish_listener_status(
                status="transcribed",
                message=f"Texte entendu : {transcription}",
                is_active=True,
                transcription=transcription,
            )

            waiting_for_follow_up_command, chatbot_mode = process_transcription(
                transcription=transcription,
                wake_detector=wake_detector,
                parser=parser,
                waiting_for_follow_up_command=waiting_for_follow_up_command,
                chatbot_mode=chatbot_mode,
            )

            if waiting_for_follow_up_command:
                follow_up_deadline = time.time() + FOLLOW_UP_SECONDS

            if chatbot_mode:
                print_terminal("MODE", "Chatbot actif. Les prochaines phrases iront à Ollama.")
            else:
                print_terminal("MODE", "Mode commandes dashboard.")

            print()

    except KeyboardInterrupt:
        print()
        print_terminal("STOP", "Silero VAD listener stopped.")

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