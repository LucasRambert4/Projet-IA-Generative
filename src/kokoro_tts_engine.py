from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
TTS_OUTPUT_FILE = RUNTIME_DIR / "kokoro_response.wav"


def synthesize_response_to_wav(text: str) -> str:
    """
    Converts Ollama's text response into a WAV audio file using Kokoro TTS.

    Returns the path to the generated audio file.
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    # TODO:
    # Replace this section with the real Kokoro TTS call.
    #
    # Expected behavior:
    # - input: text
    # - output: runtime/kokoro_response.wav

    raise NotImplementedError(
        "Kokoro TTS integration must be connected here."
    )

    return str(TTS_OUTPUT_FILE)