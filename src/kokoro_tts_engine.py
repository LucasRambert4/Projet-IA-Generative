import re
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass

from kokoro import KPipeline


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*dropout option adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*weight_norm.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*unauthenticated requests to the HF Hub.*",
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
TTS_OUTPUT_FILE = RUNTIME_DIR / "kokoro_response.wav"

SAMPLE_RATE = 24000
LANG_CODE = "f"
VOICE = "ff_siwis"
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"

_pipeline: KPipeline | None = None


def _get_pipeline() -> KPipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = KPipeline(lang_code=LANG_CODE, repo_id=KOKORO_REPO_ID)

    return _pipeline


def prepare_text_for_tts(text: str, max_chars: int = 600) -> str:
    """
    Cleans assistant text before speech synthesis.
    """

    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)
    cleaned = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    if len(cleaned) > max_chars:
        truncated = cleaned[:max_chars]
        last_space = truncated.rfind(" ")

        if last_space > max_chars // 2:
            truncated = truncated[:last_space]

        cleaned = truncated.rstrip(" ,;:") + "."

    return cleaned


def synthesize_response_to_wav(text: str) -> str:
    """
    Converts Ollama's text response into a WAV audio file using Kokoro TTS.

    Returns the path to the generated audio file.
    """

    prepared_text = prepare_text_for_tts(text)

    if not prepared_text:
        raise ValueError("Le texte à synthétiser est vide.")

    RUNTIME_DIR.mkdir(exist_ok=True)

    pipeline = _get_pipeline()
    generator = pipeline(prepared_text, voice=VOICE)

    audio_parts = [audio for _, _, audio in generator if audio is not None]

    if not audio_parts:
        raise RuntimeError("Kokoro n'a généré aucun audio.")

    full_audio = np.concatenate(audio_parts)
    sf.write(TTS_OUTPUT_FILE, full_audio, SAMPLE_RATE)

    return str(TTS_OUTPUT_FILE)
