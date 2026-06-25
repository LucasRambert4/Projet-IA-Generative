from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
TTS_OUTPUT_FILE = RUNTIME_DIR / "kokoro_response.wav"

KOKORO_SAMPLE_RATE = 24000
KOKORO_LANG_CODE = "f"
KOKORO_VOICE = "ff_siwis"
KOKORO_SPEED = 1.08

MAX_TTS_CHARACTERS = 320

_pipeline = None


def get_pipeline():
    """
    Loads Kokoro only when TTS is actually needed.

    This avoids slowing down Streamlit startup.
    """

    global _pipeline

    if _pipeline is None:
        from kokoro import KPipeline

        _pipeline = KPipeline(
            lang_code=KOKORO_LANG_CODE,
            repo_id="hexgrad/Kokoro-82M",
        )

    return _pipeline


def clean_text_for_tts(text: str) -> str:
    """
    Cleans assistant response before speech synthesis.

    The text is intentionally shortened because long TTS generation is slow
    and not useful for a voice dashboard demo.
    """

    if not text:
        return ""

    cleaned_text = text.strip()

    replacements = {
        "*": "",
        "#": "",
        "`": "",
        "|": " ",
        "- ": "",
        "•": "",
        "\n": " ",
    }

    for old_value, new_value in replacements.items():
        cleaned_text = cleaned_text.replace(old_value, new_value)

    cleaned_text = " ".join(cleaned_text.split())

    if len(cleaned_text) <= MAX_TTS_CHARACTERS:
        return cleaned_text

    shortened_text = cleaned_text[:MAX_TTS_CHARACTERS]

    sentence_endings = [".", "!", "?"]

    last_sentence_position = -1

    for ending in sentence_endings:
        last_sentence_position = max(
            last_sentence_position,
            shortened_text.rfind(ending),
        )

    if last_sentence_position > 80:
        shortened_text = shortened_text[:last_sentence_position + 1]
    else:
        shortened_text = shortened_text.rstrip() + "."

    return shortened_text


def synthesize_response_to_wav(text: str) -> str | None:
    """
    Converts chatbot response text into a WAV file.

    Heavy imports are inside this function to keep Streamlit startup fast.
    """

    if not text:
        return None

    import numpy as np
    import soundfile as sf

    RUNTIME_DIR.mkdir(exist_ok=True)

    cleaned_text = clean_text_for_tts(text)

    if not cleaned_text:
        return None

    pipeline = get_pipeline()

    generator = pipeline(
        cleaned_text,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
    )

    audio_chunks = []

    for _, _, audio in generator:
        audio_chunks.append(audio)

    if not audio_chunks:
        return None

    combined_audio = np.concatenate(audio_chunks)

    sf.write(
        TTS_OUTPUT_FILE,
        combined_audio,
        KOKORO_SAMPLE_RATE,
    )

    return str(TTS_OUTPUT_FILE)