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
TEMP_AUDIO_FILE = RUNTIME_DIR / "listener_chunk.wav"

SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 4
SILENCE_THRESHOLD = 0.006

MODEL_SIZE = "base"

FOLLOW_UP_SECONDS = 6


def record_audio_chunk() -> bool:
    """
    Records a short audio chunk from the local microphone.

    Returns True if the audio is loud enough to be processed.
    Returns False if the chunk is probably silence.
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    audio = sd.rec(
        int(CHUNK_DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    energy = float(np.sqrt(np.mean(audio ** 2)))

    if energy < SILENCE_THRESHOLD:
        return False

    sf.write(TEMP_AUDIO_FILE, audio, SAMPLE_RATE)

    return True


def main():
    print("Loading local STT model...")
    stt = LocalSTTEngine(model_size=MODEL_SIZE)

    wake_detector = WakeWordDetector()
    parser = CommandParser()

    waiting_for_follow_up_command = False
    follow_up_deadline = 0

    print("\nContinuous voice listener started.")
    print("Say: Ok Jack, affiche les ventes par région")
    print("Press CTRL + C to stop.\n")

    while True:
        try:
            has_voice = record_audio_chunk()

            if not has_voice:
                continue

            transcription = stt.transcribe_audio(
                str(TEMP_AUDIO_FILE),
                language="fr"
            )

            transcription = transcription.strip()

            if not transcription:
                continue

            print(f"Transcription: {transcription}")

            wake_result = wake_detector.extract_command(transcription)
            now = time.time()

            if wake_result["activated"]:
                command_text = wake_result["command_text"]

                if command_text:
                    parsed_command = parser.parse(command_text)

                    write_command_event(
                        transcription=transcription,
                        wake_result=wake_result,
                        parsed_command=parsed_command
                    )

                    waiting_for_follow_up_command = False

                    print("Wake word detected.")
                    print(f"Command: {command_text}")
                    print(f"Parsed: {parsed_command}\n")

                else:
                    waiting_for_follow_up_command = True
                    follow_up_deadline = now + FOLLOW_UP_SECONDS

                    print("Wake word detected. Waiting for the next command...\n")

            elif waiting_for_follow_up_command and now <= follow_up_deadline:
                command_text = transcription
                parsed_command = parser.parse(command_text)

                wake_result = {
                    "activated": True,
                    "wake_word": "ok jack",
                    "command_text": command_text,
                    "original_text": transcription,
                    "mode": "follow_up_command"
                }

                write_command_event(
                    transcription=transcription,
                    wake_result=wake_result,
                    parsed_command=parsed_command
                )

                waiting_for_follow_up_command = False

                print("Follow-up command received.")
                print(f"Command: {command_text}")
                print(f"Parsed: {parsed_command}\n")

            elif waiting_for_follow_up_command and now > follow_up_deadline:
                waiting_for_follow_up_command = False
                print("Follow-up command timeout.\n")

        except KeyboardInterrupt:
            print("\nVoice listener stopped.")
            break

        except Exception as error:
            print(f"Error: {error}")
            time.sleep(1)


if __name__ == "__main__":
    main()