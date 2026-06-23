from stt_engine import LocalSTTEngine
from command_parser import CommandParser


def main():
    audio_path = "audio_samples/test.wav"

    print("Loading local STT model...")
    stt = LocalSTTEngine(model_size="base")

    print("Transcribing audio...")
    text = stt.transcribe_audio(audio_path, language="fr")

    print("\n--- Transcription result ---")
    print(text)

    parser = CommandParser()
    command = parser.parse(text)

    print("\n--- Parsed command ---")
    print(command)


if __name__ == "__main__":
    main()