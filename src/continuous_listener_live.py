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
from command_bus import write_command_event


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
TEMP_AUDIO_FILE = RUNTIME_DIR / "listener_phrase.wav"


# ==========================================================
# FIXED MICROPHONE CONFIGURATION
# ==========================================================

FIXED_INPUT_DEVICE = 1
FIXED_SAMPLE_RATE = 44100
FIXED_SPEECH_THRESHOLD = 0.008


# ==========================================================
# RECORDING CONFIGURATION
# ==========================================================

FRAME_DURATION_SECONDS = 0.25
END_SILENCE_SECONDS = 0.9
MIN_SPEECH_SECONDS = 0.7
MAX_UTTERANCE_SECONDS = 8

FOLLOW_UP_SECONDS = 3
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")


def print_terminal(label: str, message: str = ""):
    if message:
        print(f"[{label}] {message}", flush=True)
    else:
        print(f"[{label}]", flush=True)


class PersistentMicrophone:
    """
    Keeps the microphone stream open permanently.

    We use int16 for better Windows/Realtek compatibility,
    then convert audio to float32 internally.
    """

    def __init__(self, input_device: int, sample_rate: int):
        self.input_device = input_device
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print_terminal("AUDIO WARNING", str(status))

        # Convert int16 audio to float32 between -1 and 1.
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
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    frames = []
    is_recording = False
    silence_duration = 0.0
    speech_duration = 0.0
    max_energy = 0.0
    start_time = None

    while True:
        frame, energy = microphone.read_frame()
        max_energy = max(max_energy, energy)

        if not is_recording:
            print(
                f"\r[WAITING] niveau micro {energy:.6f} | seuil {speech_threshold:.6f}",
                end="",
                flush=True,
            )

            if energy >= speech_threshold:
                is_recording = True
                start_time = time.time()
                frames.append(frame)
                speech_duration += FRAME_DURATION_SECONDS
                silence_duration = 0.0

                print()
                print_terminal("RECORDING", "Voix détectée, enregistrement de la phrase...")

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
        return None, speech_duration, max_energy

    audio = np.concatenate(frames, axis=0)
    audio = normalize_audio(audio)

    sf.write(TEMP_AUDIO_FILE, audio, sample_rate)

    return str(TEMP_AUDIO_FILE), speech_duration, max_energy


def process_transcription(
    transcription: str,
    wake_detector: WakeWordDetector,
    parser: CommandParser,
    waiting_for_follow_up_command: bool,
) -> bool:
    transcription = transcription.strip()

    if not transcription:
        print_terminal("TEXT", "Transcription vide.")
        print()
        return waiting_for_follow_up_command

    print_terminal("TEXT", transcription)

    wake_result = wake_detector.extract_command(transcription)

    if wake_result["activated"]:
        command_text = wake_result["command_text"]

        print_terminal("WAKE", "Ok Jack détecté.")

        if command_text:
            parsed_command = parser.parse(command_text)

            write_command_event(
                transcription=transcription,
                wake_result=wake_result,
                parsed_command=parsed_command,
            )

            print_terminal("COMMAND", command_text)
            print_terminal("PARSED", str(parsed_command))
            print()

            return False

        print_terminal(
            "SESSION",
            f"Ok Jack détecté. En attente de la commande pendant {FOLLOW_UP_SECONDS}s."
        )
        print()

        return True

    if waiting_for_follow_up_command:
        parsed_command = parser.parse(transcription)

        if parsed_command.get("intent") == "unknown":
            print_terminal("FOLLOW-UP", transcription)
            print_terminal("ERROR", "Commande non reconnue, réessayez.")
            print()
            return True

        wake_result = {
            "activated": True,
            "wake_word": "ok jack",
            "command_text": transcription,
            "original_text": transcription,
            "mode": "follow_up_command",
        }

        write_command_event(
            transcription=transcription,
            wake_result=wake_result,
            parsed_command=parsed_command,
        )

        print_terminal("FOLLOW-UP", "Commande reçue après Ok Jack.")
        print_terminal("COMMAND", transcription)
        print_terminal("PARSED", str(parsed_command))
        print()

        return False

    print_terminal("IGNORED", "Pas de wake word, phrase ignorée.")
    print()

    return waiting_for_follow_up_command


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

    print_terminal("START", f"Chargement de Whisper model: {MODEL_SIZE}")
    stt = LocalSTTEngine(model_size=MODEL_SIZE)

    wake_detector = WakeWordDetector()
    parser = CommandParser()

    waiting_for_follow_up_command = False
    follow_up_deadline = 0.0

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
    print_terminal("EXAMPLE", "Say: Ok Jack, affiche les ventes par région")
    print_terminal("STOP", "Press CTRL + C to stop.")
    print()

    try:
        microphone.start()
        print_terminal("MIC", "Microphone stream opened and kept alive.")
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

            transcription = stt.transcribe_audio(
                audio_path,
                language="fr"
            ).strip()

            now = time.time()

            waiting_for_follow_up_command = process_transcription(
                transcription,
                wake_detector,
                parser,
                waiting_for_follow_up_command,
            )

            if waiting_for_follow_up_command:
                follow_up_deadline = now + FOLLOW_UP_SECONDS

    except KeyboardInterrupt:
        print()
        print_terminal("STOP", "Phrase listener stopped.")

    except Exception as error:
        print()
        print_terminal("ERROR", str(error))

    finally:
        microphone.stop()
        print_terminal("MIC", "Microphone stream closed.")


if __name__ == "__main__":
    main()