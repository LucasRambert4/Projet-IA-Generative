"""Navigation vocale locale : STT (faster-whisper) + interpretation Ollama (LLM)."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_STT_MODEL = os.getenv("OLLAMA_STT_MODEL", "small")
STT_INITIAL_PROMPT = (
    "OK Google, navigation dashboard, compagnies, aeroports, routes, temporalite, "
    "filtre, mois, janvier, va a, affiche."
)
VOICE_STT_FAST_BEAM_SIZE = int(os.getenv("VOICE_STT_FAST_BEAM_SIZE", "3"))
VOICE_STT_BEAM_SIZE = int(os.getenv("VOICE_STT_BEAM_SIZE", "5"))
VOICE_STT_MIN_AUDIO_SECONDS = float(os.getenv("VOICE_STT_MIN_AUDIO_SECONDS", "1.2"))
WAKE_WORD_LABEL = "OK Google"
# Variantes STT courantes : « au google », « ok gogle », « ok goggle », etc.
WAKE_WORD_SEARCH = re.compile(
    r"(?:ok(?:ay)?|au|o)\s*[- ]?\s*"
    r"(?:g[\s-]*)?(?:google|goggle|gogle|goulg|googol)"
    r"\s*[,;.\-:]?\s*",
    re.IGNORECASE,
)
WAKE_WORD_PATTERN = WAKE_WORD_SEARCH
WAKE_WORD_COMPACT_ALIASES = (
    "okgoogle",
    "okaygoogle",
    "augoogle",
    "okgoggle",
    "okgogle",
    "okgoulg",
)
VOICE_SAMPLE_RATE = 16_000
VOICE_SILENCE_SECONDS = float(os.getenv("VOICE_SILENCE_SECONDS", "5"))
VOICE_SILENCE_THRESHOLD = float(os.getenv("VOICE_SILENCE_THRESHOLD", "0.006"))
VOICE_MAX_RECORD_SECONDS = float(os.getenv("VOICE_MAX_RECORD_SECONDS", "30"))
VOICE_WAKE_WAIT_SECONDS = float(os.getenv("VOICE_WAKE_WAIT_SECONDS", "45"))
VOICE_CONTINUOUS_CYCLE_SECONDS = float(os.getenv("VOICE_CONTINUOUS_CYCLE_SECONDS", "90"))
VOICE_CHUNK_SECONDS = 0.1
VOICE_PARTIAL_TRANSCRIBE_SECONDS = float(
    os.getenv("VOICE_PARTIAL_TRANSCRIBE_SECONDS", "2")
)
VIRTUAL_DEVICE_KEYWORDS = (
    "instashare",
    "zoom",
    "teams audio",
    "blackhole",
    "soundflower",
    "aggregate",
    "loopback",
    "parrot",
    "virtual",
    "vb-audio",
    "vb audio",
)
_runtime: dict[str, int | None] = {"input_device": None}

__all__ = [
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "OLLAMA_STT_MODEL",
    "WAKE_WORD_LABEL",
    "VOICE_SAMPLE_RATE",
    "VOICE_SILENCE_SECONDS",
    "VOICE_SILENCE_THRESHOLD",
    "VOICE_MAX_RECORD_SECONDS",
    "VOICE_CHUNK_SECONDS",
    "TAB_LABELS",
    "VoiceCommand",
    "check_ollama_available",
    "load_stt_model",
    "transcribe_audio",
    "transcribe_chunks",
    "record_until_silence",
    "list_input_devices",
    "resolve_input_device",
    "set_input_device",
    "get_input_device",
    "input_device_label",
    "run_alexa_voice_loop",
    "contains_wake_word",
    "listen_for_wake_word_once",
    "listen_for_wake_word_and_command",
    "WakeListenResult",
    "extract_command_after_wake_word",
    "parse_voice_command",
    "parse_voice_command_ollama",
    "parse_voice_command_rules",
    "process_voice_transcript",
    "apply_voice_command",
    "init_voice_session_state",
    "generate_data_synthesis",
]

TAB_LABELS = [
    "1. Vue globale",
    "2. Temporalite",
    "3. Aeroports",
    "4. Compagnies",
    "5. Même avion",
    "6. Routes",
    "7. Decisions",
    "8. Modelisation",
    "9. Conclusion",
]

MONTH_NAMES: dict[str, int] = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}

Action = Literal[
    "navigate",
    "filter_airline",
    "filter_origin",
    "filter_destination",
    "filter_month",
    "clear_filters",
    "read_kpi",
    "unknown",
]

VALID_ACTIONS: set[str] = {
    "navigate",
    "filter_airline",
    "filter_origin",
    "filter_destination",
    "filter_month",
    "clear_filters",
    "read_kpi",
    "unknown",
}


@dataclass
class VoiceCommand:
    action: Action
    tab: str | None = None
    value: str | int | None = None
    message: str = ""


class OllamaUnavailableError(RuntimeError):
    """Levee quand le serveur ou le modele Ollama est indisponible."""


@dataclass
class WakeListenResult:
    """Resultat d'une tentative d'ecoute du mot d'activation."""

    transcript: str | None = None
    last_heard: str | None = None
    heard_speech: bool = False
    peak_rms: float = 0.0


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _compact_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


def contains_wake_word(text: str) -> bool:
    """Detecte le mot d'activation n'importe ou dans la transcription."""
    if WAKE_WORD_SEARCH.search(text or ""):
        return True
    compact = _compact_alnum(text or "")
    return any(alias in compact for alias in WAKE_WORD_COMPACT_ALIASES)


def extract_command_after_wake_word(text: str) -> tuple[str | None, str]:
    """Verifie le mot d'activation « OK Google » et retourne la commande utile."""
    if not text or not text.strip():
        return None, "Aucune commande detectee."

    match = WAKE_WORD_SEARCH.search(text)
    if not match and contains_wake_word(text):
        compact = _compact_alnum(text)
        for alias in WAKE_WORD_COMPACT_ALIASES:
            start = compact.find(alias)
            if start >= 0:
                ratio = len(text) / max(len(compact), 1)
                approx_end = min(len(text), int((start + len(alias)) * ratio) + 2)
                command = text[approx_end:].strip(" ,;.-:")
                if command:
                    return command, ""
                return None, (
                    f"« {WAKE_WORD_LABEL} » detecte : ajoutez une commande "
                    "(ex. « va a compagnies »)."
                )

    if not match:
        return None, (
            f"Mot d'activation requis : dites « {WAKE_WORD_LABEL} » puis la commande. "
            f"Ex. : « {WAKE_WORD_LABEL}, va a compagnies »."
        )

    command = text[match.end() :].strip()
    if not command:
        return None, (
            f"« {WAKE_WORD_LABEL} » detecte : ajoutez une commande "
            "(ex. « va a compagnies »)."
        )

    return command, ""


def check_ollama_available(model: str | None = None) -> bool:
    """Verifie que Ollama repond et que le modele LLM est present."""
    try:
        import ollama

        client = ollama.Client(host=OLLAMA_HOST)
        tags = client.list()
        models = [m.model for m in tags.models]
        target = model or OLLAMA_MODEL
        base = target.split(":")[0]
        return any(m == target or m.startswith(f"{base}:") for m in models)
    except Exception:
        return False


def load_stt_model(model_size: str | None = None):
    """Charge faster-whisper (defaut : small, bon compromis qualite/francais sur CPU)."""
    from faster_whisper import WhisperModel

    size = model_size or OLLAMA_STT_MODEL
    return WhisperModel(
        size,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(2, os.cpu_count() or 2),
        num_workers=1,
    )


def _frame_to_float_mono(frame) -> "np.ndarray":
    import numpy as np

    array = frame.to_ndarray()
    if array.ndim > 1:
        array = array.mean(axis=0)
    array = array.reshape(-1)
    if array.dtype == np.int16:
        return (array.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if array.dtype == np.int32:
        return (array.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    return array.astype(np.float32)


def decode_audio_bytes(data: bytes) -> tuple["np.ndarray", int]:
    """Decode un fichier audio (WAV, WebM, MP4...) via PyAV."""
    import io

    import av
    import numpy as np

    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.audio:
            return np.array([], dtype=np.float32), VOICE_SAMPLE_RATE
        stream = container.streams.audio[0]
        sample_rate = int(stream.sample_rate or VOICE_SAMPLE_RATE)
        parts: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            parts.append(_frame_to_float_mono(frame))
        if not parts:
            return np.array([], dtype=np.float32), sample_rate
        return np.concatenate(parts), sample_rate


def resample_audio_pyav(
    audio: "np.ndarray",
    source_rate: int,
    target_rate: int = VOICE_SAMPLE_RATE,
) -> "np.ndarray":
    """Reechantillonnage audio fiable via PyAV (mono float32)."""
    import av
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or source_rate == target_rate:
        return samples

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = source_rate

    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_rate)
    output: list[np.ndarray] = []
    for resampled in resampler.resample(frame):
        output.append(_frame_to_float_mono(resampled))
    for resampled in resampler.resample(None):
        output.append(_frame_to_float_mono(resampled))

    if not output:
        return np.array([], dtype=np.float32)
    return np.concatenate(output).astype(np.float32)


def prepare_audio_for_whisper(
    source: bytes | "np.ndarray",
    sample_rate: int | None = None,
) -> "np.ndarray":
    """Prepare un signal float32 mono 16 kHz pour faster-whisper."""
    import numpy as np

    if isinstance(source, np.ndarray):
        audio = source.astype(np.float32).reshape(-1)
        rate = sample_rate or VOICE_SAMPLE_RATE
    else:
        audio, rate = decode_audio_bytes(source)

    if audio.size == 0:
        return np.array([], dtype=np.float32)
    return resample_audio_pyav(audio, rate, VOICE_SAMPLE_RATE)


def transcribe_chunks(
    chunks: list,
    capture_rate: int,
    model=None,
    *,
    fast: bool = False,
) -> str:
    """Transcrit des blocs micro (float32) via faster-whisper."""
    import numpy as np

    if not chunks:
        return ""
    audio = prepare_audio_for_whisper(np.concatenate(chunks), capture_rate)
    if audio.size == 0:
        return ""
    return transcribe_audio(audio, model=model, fast=fast)


def _chunks_duration_seconds(chunks: list, sample_rate: int) -> float:
    import numpy as np

    if not chunks or sample_rate <= 0:
        return 0.0
    return sum(len(np.asarray(c).reshape(-1)) for c in chunks) / sample_rate


def transcribe_audio(
    audio: bytes | "np.ndarray",
    model=None,
    *,
    sample_rate: int | None = None,
    fast: bool = False,
) -> str:
    """Transcrit l'audio via faster-whisper (numpy 16 kHz ou bytes via PyAV)."""
    import numpy as np

    if model is None:
        model = load_stt_model()

    if isinstance(audio, np.ndarray):
        samples = audio.astype(np.float32).reshape(-1)
        if sample_rate and sample_rate != VOICE_SAMPLE_RATE:
            samples = prepare_audio_for_whisper(samples, sample_rate)
    else:
        samples = prepare_audio_for_whisper(audio)

    if samples.size == 0:
        return ""

    segments, _ = model.transcribe(
        samples,
        language="fr",
        task="transcribe",
        initial_prompt=STT_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 200,
        },
        beam_size=VOICE_STT_FAST_BEAM_SIZE if fast else VOICE_STT_BEAM_SIZE,
        best_of=1 if fast else VOICE_STT_BEAM_SIZE,
        hotwords="OK Google, compagnies, aeroports, routes, temporalite, dashboard",
        length_penalty=1.0,
        repetition_penalty=1.1,
        no_speech_threshold=0.55,
        log_prob_threshold=-0.8,
        compression_ratio_threshold=2.4,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    return " ".join(segment.text for segment in segments).strip()


def list_input_devices() -> list[dict[str, int | str]]:
    """Liste les peripheriques d'entree audio disponibles."""
    import sounddevice as sd

    devices: list[dict[str, int | str]] = []
    for index, info in enumerate(sd.query_devices()):
        max_in = int(info["max_input_channels"])
        if max_in < 1:
            continue
        devices.append(
            {
                "index": index,
                "name": str(info["name"]),
                "sample_rate": int(info["default_samplerate"]),
                "channels": max_in,
            }
        )
    return devices


def _score_input_device(name: str, channels: int) -> int:
    lower = name.lower()
    score = 0
    if any(keyword in lower for keyword in VIRTUAL_DEVICE_KEYWORDS):
        score -= 1000
    if "microphone" in lower or "micro " in lower:
        score += 60
    if "macbook" in lower or "built-in" in lower or "built in" in lower:
        score += 40
    if channels == 1:
        score += 20
    return score


def resolve_input_device(device: int | None = None) -> int:
    """Choisit un peripherique micro fiable (evite les drivers virtuels type InstaShare/Zoom)."""
    import sounddevice as sd

    if device is not None:
        return int(device)

    if _runtime.get("input_device") is not None:
        return int(_runtime["input_device"])

    env = os.getenv("VOICE_INPUT_DEVICE", "").strip()
    if env:
        if env.isdigit():
            return int(env)
        for item in list_input_devices():
            if env.lower() in str(item["name"]).lower():
                return int(item["index"])

    best_index: int | None = None
    best_score = -10_000
    for item in list_input_devices():
        score = _score_input_device(str(item["name"]), int(item["channels"]))
        if score > best_score:
            best_score = score
            best_index = int(item["index"])

    if best_index is None:
        raise RuntimeError("Aucun peripherique d'entree audio detecte.")

    # Evite le defaut systeme souvent pointe vers InstaShare/Zoom (silencieux).
    try:
        default_in = sd.default.device[0]
        if default_in >= 0 and best_score < 0:
            return int(default_in)
    except Exception:
        pass

    return best_index


def set_input_device(device: int | None) -> None:
    _runtime["input_device"] = int(device) if device is not None else None


def get_input_device() -> int:
    return resolve_input_device(_runtime.get("input_device"))


def input_device_label(device: int | None = None) -> str:
    import sounddevice as sd

    index = resolve_input_device(device)
    return str(sd.query_devices(index)["name"])


def _input_channels(device: int) -> int:
    import sounddevice as sd

    return min(1, int(sd.query_devices(device)["max_input_channels"])) or 1


def _device_stream_config(device: int) -> tuple[int, int, int]:
    """Retourne (device, channels, sample_rate) compatibles macOS/CoreAudio."""
    import sounddevice as sd

    info = sd.query_devices(device)
    channels = _input_channels(device)
    sample_rate = int(info["default_samplerate"])
    return int(device), channels, sample_rate


def _threshold_from_chunks(chunks: list, fallback: float = VOICE_SILENCE_THRESHOLD) -> float:
    import numpy as np

    if not chunks:
        return fallback
    levels = [
        float(np.sqrt(np.mean(np.square(np.asarray(chunk, dtype=np.float32)))))
        for chunk in chunks
    ]
    ambient = float(np.median(levels))
    return max(0.003, min(0.03, max(fallback, ambient * 3.0)))


def _record_chunk(
    frames: int,
    sample_rate: int,
    device: int,
) -> object:
    """Capture un bloc audio mono depuis le peripherique choisi."""
    import numpy as np
    import sounddevice as sd

    channels = _input_channels(device)
    chunk = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    if chunk.ndim > 1 and chunk.shape[1] > 1:
        chunk = np.mean(chunk, axis=1, keepdims=True)
    return chunk


def _adaptive_speech_threshold(
    sample_rate: int,
    chunk_samples: int,
    input_device: int | None = None,
) -> float:
    """Calibre le seuil VAD selon le bruit ambiant du micro."""
    import numpy as np

    device = resolve_input_device(input_device)
    levels: list[float] = []
    for _ in range(max(1, int(0.4 / VOICE_CHUNK_SECONDS))):
        chunk = _record_chunk(chunk_samples, sample_rate, device)
        levels.append(float(np.sqrt(np.mean(np.square(chunk)))))

    ambient = float(np.median(levels)) if levels else VOICE_SILENCE_THRESHOLD
    return max(0.003, min(0.025, max(VOICE_SILENCE_THRESHOLD, ambient * 3.5)))


def _chunks_to_wav_bytes(chunks: list, sample_rate: int) -> bytes:
    """Convertit des morceaux numpy en WAV mono 16 kHz."""
    import io
    import wave

    import numpy as np

    if not chunks:
        return b""
    audio = np.concatenate(chunks, axis=0)
    pcm = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _float_audio_to_wav_bytes(
    audio,
    sample_rate: int = VOICE_SAMPLE_RATE,
) -> bytes:
    """Convertit un signal float32 mono en WAV."""
    import io
    import wave

    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    pcm = (samples * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _calibrate_threshold(device: int, sample_rate: int, frames: int = 1024) -> float:
    """Mesure le bruit ambiant pour le VAD."""
    import time

    import numpy as np

    levels: list[float] = []
    deadline = time.time() + 0.35
    while time.time() < deadline:
        chunk = np.asarray(_record_chunk(frames, sample_rate, device), dtype=np.float32)
        levels.append(float(np.sqrt(np.mean(np.square(chunk)))))
    ambient = float(np.median(levels)) if levels else VOICE_SILENCE_THRESHOLD
    return max(0.003, min(0.03, max(VOICE_SILENCE_THRESHOLD, ambient * 3.0)))


def run_alexa_voice_loop(
    event_queue,
    get_device: Callable[[], int],
    stt_model=None,
) -> None:
    """
    Ecoute continue type Alexa :
    1. Attente du mot d'activation « OK Google »
    2. Capture de la commande avec transcription live
    3. Retour en ecoute
    """
    import queue as thread_queue
    import time

    import numpy as np
    import sounddevice as sd

    if stt_model is None:
        stt_model = load_stt_model()

    blocksize = 1024
    wake_window_seconds = 4.0
    wake_scan_seconds = 1.8
    command_silence_seconds = min(2.5, VOICE_SILENCE_SECONDS)
    command_live_seconds = 1.2
    min_transcribe_seconds = VOICE_STT_MIN_AUDIO_SECONDS
    calibrate_chunks = 8

    def emit(event_type: str, **payload) -> None:
        event_queue.put({"type": event_type, **payload})

    while True:
        device, channels, capture_rate = _device_stream_config(get_device())
        chunk_duration = blocksize / capture_rate
        audio_queue: thread_queue.Queue = thread_queue.Queue(maxsize=256)

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                return
            if indata.ndim > 1:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata.reshape(-1)
            try:
                audio_queue.put_nowait(np.asarray(mono, dtype=np.float32).copy())
            except thread_queue.Full:
                pass

        try:
            emit(
                "status",
                phase="wake",
                message=(
                    f"En ecoute — dites « {WAKE_WORD_LABEL} » "
                    f"({input_device_label(device)}, {capture_rate} Hz)"
                ),
            )
            emit("live", text="")

            state = "wake"
            wake_chunks: list[np.ndarray] = []
            command_chunks: list[np.ndarray] = []
            silence_seconds = 0.0
            last_scan = 0.0
            threshold = VOICE_SILENCE_THRESHOLD
            calibrated = False
            warmup: list[np.ndarray] = []

            with sd.InputStream(
                device=device,
                channels=channels,
                samplerate=capture_rate,
                dtype="float32",
                blocksize=blocksize,
                latency="high",
                callback=callback,
            ):
                while True:
                    if get_device() != device:
                        break

                    try:
                        chunk = audio_queue.get(timeout=0.8)
                    except thread_queue.Empty:
                        continue

                    chunk = chunk.reshape(-1)

                    if not calibrated:
                        warmup.append(chunk)
                        if len(warmup) >= calibrate_chunks:
                            threshold = _threshold_from_chunks(warmup, threshold)
                            calibrated = True
                            warmup = []
                        continue

                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    has_speech = rms >= threshold

                    if state == "wake":
                        if has_speech:
                            wake_chunks.append(chunk)
                            max_samples = int(wake_window_seconds * capture_rate)
                            total = sum(len(item) for item in wake_chunks)
                            while total > max_samples and wake_chunks:
                                total -= len(wake_chunks.pop(0))

                            now = time.time()
                            if wake_chunks and now - last_scan >= wake_scan_seconds:
                                last_scan = now
                                if (
                                    _chunks_duration_seconds(wake_chunks, capture_rate)
                                    >= min_transcribe_seconds
                                ):
                                    text = transcribe_chunks(
                                        wake_chunks,
                                        capture_rate,
                                        model=stt_model,
                                        fast=True,
                                    )
                                    if text:
                                        emit("live", text=text)
                                        if contains_wake_word(text):
                                            emit(
                                                "status",
                                                phase="command",
                                                message=(
                                                    "OK Google detecte — "
                                                    "parlez votre commande."
                                                ),
                                            )
                                            state = "command"
                                            command_chunks = list(wake_chunks)
                                            wake_chunks = []
                                            silence_seconds = 0.0
                                            last_scan = now
                        elif len(wake_chunks) > 30:
                            wake_chunks = wake_chunks[-15:]

                    else:
                        command_chunks.append(chunk)
                        if has_speech:
                            silence_seconds = 0.0
                        else:
                            silence_seconds += chunk_duration

                        now = time.time()
                        if (
                            has_speech
                            and command_chunks
                            and now - last_scan >= command_live_seconds
                            and _chunks_duration_seconds(command_chunks, capture_rate)
                            >= min_transcribe_seconds
                        ):
                            last_scan = now
                            text = transcribe_chunks(
                                command_chunks,
                                capture_rate,
                                model=stt_model,
                                fast=True,
                            )
                            if text:
                                emit("live", text=text)

                        if silence_seconds >= command_silence_seconds and command_chunks:
                            full_text = transcribe_chunks(
                                command_chunks,
                                capture_rate,
                                model=stt_model,
                                fast=False,
                            )
                            emit("live", text=full_text or "")
                            if full_text and contains_wake_word(full_text):
                                emit("command", text=full_text)
                            elif full_text:
                                emit(
                                    "error",
                                    message=(
                                        f"Commande sans « {WAKE_WORD_LABEL} ». "
                                        "Recommencez."
                                    ),
                                )
                            state = "wake"
                            command_chunks = []
                            wake_chunks = []
                            silence_seconds = 0.0
                            last_scan = 0.0
                            emit(
                                "status",
                                phase="wake",
                                message=(
                                    f"En ecoute — dites « {WAKE_WORD_LABEL} » "
                                    f"({input_device_label(device)}, {capture_rate} Hz)"
                                ),
                            )

        except Exception as exc:
            emit("error", message=str(exc))
            emit(
                "status",
                phase="error",
                message="Micro en erreur — changez de peripherique dans la liste.",
            )
            time.sleep(1.5)


def record_until_silence(
    silence_seconds: float | None = None,
    sample_rate: int = VOICE_SAMPLE_RATE,
    max_duration: float = VOICE_MAX_RECORD_SECONDS,
    wait_for_speech_first: bool = False,
    stats: dict[str, float | bool] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    stt_model=None,
    input_device: int | None = None,
) -> bytes:
    """
    Enregistre le micro jusqu'a N secondes de silence consecutif apres la parole.

    Si wait_for_speech_first=True, ignore le bruit ambiant jusqu'a detecter une voix.
    """
    import io
    import wave

    import numpy as np

    device = resolve_input_device(input_device)
    silence_seconds = silence_seconds or VOICE_SILENCE_SECONDS
    chunk_samples = max(1, int(sample_rate * VOICE_CHUNK_SECONDS))
    silence_chunks_needed = max(1, int(silence_seconds / VOICE_CHUNK_SECONDS))
    effective_max = max_duration + (VOICE_WAKE_WAIT_SECONDS if wait_for_speech_first else 0)
    max_chunks = max(1, int(effective_max / VOICE_CHUNK_SECONDS))

    recorded: list = []
    silent_chunks = 0
    heard_speech = False
    peak_rms = 0.0
    speech_threshold = (
        _adaptive_speech_threshold(sample_rate, chunk_samples, input_device=device)
        if wait_for_speech_first
        else VOICE_SILENCE_THRESHOLD
    )
    partial_every = max(
        1, int(VOICE_PARTIAL_TRANSCRIBE_SECONDS / VOICE_CHUNK_SECONDS)
    )
    chunks_since_partial = 0

    if on_progress and wait_for_speech_first:
        on_progress("waiting", "")

    for _ in range(max_chunks):
        chunk = _record_chunk(chunk_samples, sample_rate, device)
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak_rms = max(peak_rms, rms)

        if rms >= speech_threshold:
            heard_speech = True
            silent_chunks = 0
            recorded.append(chunk.copy())
            if on_progress:
                on_progress("recording", "")
            if stt_model and on_progress and len(recorded) >= partial_every:
                chunks_since_partial += 1
                if chunks_since_partial >= partial_every:
                    chunks_since_partial = 0
                    try:
                        partial = transcribe_chunks(
                            recorded,
                            sample_rate,
                            model=stt_model,
                            fast=True,
                        )
                        if partial:
                            on_progress("transcribing", partial)
                    except Exception:
                        pass
            continue

        if heard_speech:
            silent_chunks += 1
            recorded.append(chunk.copy())
            if silent_chunks >= silence_chunks_needed:
                break
        elif not wait_for_speech_first:
            continue

    if not recorded:
        if stats is not None:
            stats["peak_rms"] = peak_rms
            stats["heard_speech"] = heard_speech
            stats["speech_threshold"] = speech_threshold
        return b""

    audio = np.concatenate(recorded, axis=0)
    pcm = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    if stats is not None:
        stats["peak_rms"] = peak_rms
        stats["heard_speech"] = heard_speech
        stats["speech_threshold"] = speech_threshold
        stats["input_device"] = device
    return buffer.getvalue()


def listen_for_wake_word_once(
    stt_model=None,
    on_progress: Callable[[str, str], None] | None = None,
    input_device: int | None = None,
) -> WakeListenResult:
    """
    Une tentative d'ecoute : attend la voix, transcrit, detecte « OK Google ».

    on_progress(status, text) est appele pendant l'attente et la transcription partielle.
    """
    stats: dict[str, float | bool] = {}
    device = resolve_input_device(input_device)
    audio = record_until_silence(
        wait_for_speech_first=True,
        stats=stats,
        on_progress=on_progress,
        stt_model=stt_model,
        input_device=device,
    )
    peak_rms = float(stats.get("peak_rms", 0.0))
    heard_speech = bool(stats.get("heard_speech", False))

    if not audio:
        return WakeListenResult(heard_speech=heard_speech, peak_rms=peak_rms)

    if on_progress:
        on_progress("transcribing", "")

    transcript = transcribe_audio(audio, model=stt_model)
    if on_progress and transcript:
        on_progress("transcribing", transcript)

    if not transcript:
        return WakeListenResult(heard_speech=True, peak_rms=peak_rms)

    if contains_wake_word(transcript):
        return WakeListenResult(
            transcript=transcript,
            last_heard=transcript,
            heard_speech=True,
            peak_rms=peak_rms,
        )

    return WakeListenResult(
        last_heard=transcript,
        heard_speech=True,
        peak_rms=peak_rms,
    )


def listen_for_wake_word_and_command(
    stt_model=None,
    cycle_seconds: float | None = None,
) -> str | None:
    """
    Compatibilite : une tentative par appel (cycle_seconds ignore).

    Retourne la transcription si « OK Google » est detecte, sinon None.
    """
    _ = cycle_seconds
    return listen_for_wake_word_once(stt_model=stt_model).transcript


def _build_ollama_prompt(
    text: str,
    airline_codes: list[str] | None,
    origin_airports: list[str] | None,
    destination_airports: list[str] | None,
    airline_labels: dict[str, str] | None,
) -> tuple[str, str]:
    airline_sample = ", ".join((airline_codes or [])[:20])
    origin_sample = ", ".join((origin_airports or [])[:20])
    dest_sample = ", ".join((destination_airports or [])[:20])
    label_sample = ", ".join(
        f"{code}={label}"
        for code, label in list((airline_labels or {}).items())[:12]
    )

    system = f"""Tu interpretes des commandes vocales pour un dashboard de retards de vols.
Reponds UNIQUEMENT en JSON valide, sans markdown, avec cette structure :
{{
  "action": "navigate|filter_airline|filter_origin|filter_destination|filter_month|clear_filters|read_kpi|unknown",
  "tab": null ou un libelle d'onglet exact,
  "value": null ou une valeur (code IATA, numero de mois 1-12, ou "delay_rate"/"total" pour read_kpi),
  "message": "confirmation courte en francais"
}}

Onglets valides : {json.dumps(TAB_LABELS, ensure_ascii=False)}
Mois : janvier=1 ... decembre=12
Compagnies (codes) : {airline_sample}
Noms compagnies : {label_sample}
Aeroports depart : {origin_sample}
Aeroports arrivee : {dest_sample}

Regles :
- navigate : value=null, tab=onglet cible
- filter_* : tab=null, value=code ou mois
- clear_filters : tab=null, value=null
- read_kpi : value="delay_rate" ou "total"
- unknown si la commande n'est pas comprise"""

    return system, text.strip()


def _parse_ollama_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _normalize_ollama_command(
    payload: dict,
    airline_codes: list[str] | None,
    origin_airports: list[str] | None,
    destination_airports: list[str] | None,
    airline_labels: dict[str, str] | None,
) -> VoiceCommand:
    action = str(payload.get("action", "unknown"))
    if action not in VALID_ACTIONS:
        action = "unknown"

    tab = payload.get("tab")
    if tab is not None:
        tab = str(tab)
        if tab not in TAB_LABELS:
            match = next(
                (label for label in TAB_LABELS if normalize_text(label) in normalize_text(tab)),
                None,
            )
            tab = match

    value = payload.get("value")
    message = str(payload.get("message", "")).strip()

    if action == "filter_airline" and value is not None and airline_codes:
        code = _resolve_airline(str(value), airline_codes, airline_labels)
        if not code:
            return VoiceCommand(
                action="unknown",
                message=f"Compagnie non reconnue : {value}.",
            )
        value = code
        label = (airline_labels or {}).get(code, code)
        message = message or f"Filtre compagnie : {label}."

    if action == "filter_origin" and value is not None and origin_airports:
        airport = _match_option(str(value), origin_airports)
        if not airport:
            return VoiceCommand(action="unknown", message=f"Aeroport depart inconnu : {value}.")
        value = airport
        message = message or f"Filtre depart : {airport}."

    if action == "filter_destination" and value is not None and destination_airports:
        airport = _match_option(str(value), destination_airports)
        if not airport:
            return VoiceCommand(action="unknown", message=f"Aeroport arrivee inconnu : {value}.")
        value = airport
        message = message or f"Filtre arrivee : {airport}."

    if action == "filter_month" and value is not None:
        if isinstance(value, str) and not value.isdigit():
            month = MONTH_NAMES.get(normalize_text(value))
            if month is None:
                return VoiceCommand(action="unknown", message=f"Mois non reconnu : {value}.")
            value = month
        else:
            value = int(value)
        message = message or f"Filtre mois : {value}."

    if action == "navigate" and tab:
        message = message or f"Navigation vers {tab}."

    if action == "clear_filters":
        message = message or "Filtres reinitialises."

    return VoiceCommand(
        action=action,  # type: ignore[arg-type]
        tab=tab,
        value=value,
        message=message,
    )


def parse_voice_command_ollama(
    text: str,
    airline_codes: list[str] | None = None,
    origin_airports: list[str] | None = None,
    destination_airports: list[str] | None = None,
    airline_labels: dict[str, str] | None = None,
) -> VoiceCommand:
    """Interprete une commande via Ollama (LLM local)."""
    import ollama

    if not text or not text.strip():
        return VoiceCommand(action="unknown", message="Aucune commande detectee.")

    system, user = _build_ollama_prompt(
        text,
        airline_codes,
        origin_airports,
        destination_airports,
        airline_labels,
    )

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options={"temperature": 0},
    )
    content = response.message.content
    if not content:
        raise OllamaUnavailableError("Reponse Ollama vide.")

    payload = _parse_ollama_json(content)
    return _normalize_ollama_command(
        payload,
        airline_codes,
        origin_airports,
        destination_airports,
        airline_labels,
    )


def _resolve_tab(text: str) -> str | None:
    normalized = normalize_text(text)
    aliases = {
        "vue globale": "1. Vue globale",
        "global": "1. Vue globale",
        "temporalite": "2. Temporalite",
        "aeroports": "3. Aeroports",
        "compagnies": "4. Compagnies",
        "meme avion": "5. Même avion",
        "propagation": "5. Même avion",
        "routes": "6. Routes",
        "decisions": "7. Decisions",
        "modelisation": "8. Modelisation",
        "conclusion": "9. Conclusion",
    }
    for alias, tab in aliases.items():
        if alias in normalized:
            return tab

    match = re.search(r"onglet\s+(\d+)", normalized)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(TAB_LABELS):
            return TAB_LABELS[index]
    return None


def _extract_after_keywords(text: str, keywords: tuple[str, ...]) -> str | None:
    normalized = normalize_text(text)
    for keyword in keywords:
        if keyword in normalized:
            tail = normalized.split(keyword, 1)[1].strip(" :,-")
            if tail:
                return tail
    return None


def _match_option(value: str, options: list[str]) -> str | None:
    normalized_value = normalize_text(value)
    for option in options:
        option_norm = normalize_text(str(option))
        if normalized_value == option_norm or normalized_value in option_norm:
            return str(option)
        if option_norm in normalized_value:
            return str(option)
    return None


def _resolve_airline(
    value: str,
    airline_codes: list[str],
    airline_labels: dict[str, str] | None = None,
) -> str | None:
    direct = _match_option(value, airline_codes)
    if direct:
        return direct

    if not airline_labels:
        return None

    normalized_value = normalize_text(value)
    for code, label in airline_labels.items():
        label_norm = normalize_text(label)
        if normalized_value in label_norm or label_norm in normalized_value:
            return code
    return None


def parse_voice_command_rules(
    text: str,
    airline_codes: list[str] | None = None,
    origin_airports: list[str] | None = None,
    destination_airports: list[str] | None = None,
    airline_labels: dict[str, str] | None = None,
) -> VoiceCommand:
    """Parseur deterministe de secours si Ollama est indisponible."""
    if not text or not text.strip():
        return VoiceCommand(action="unknown", message="Aucune commande detectee.")

    normalized = normalize_text(text)

    if any(
        phrase in normalized
        for phrase in (
            "efface les filtres",
            "effacer les filtres",
            "reinitialise",
            "reinitialiser",
            "supprime les filtres",
            "tout effacer",
        )
    ):
        return VoiceCommand(action="clear_filters", message="Filtres reinitialises.")

    if any(
        phrase in normalized
        for phrase in (
            "lis le taux",
            "lire le taux",
            "quel est le taux",
            "taux de retard global",
        )
    ):
        return VoiceCommand(action="read_kpi", value="delay_rate")

    if any(
        phrase in normalized
        for phrase in ("lis les vols", "nombre de vols", "combien de vols")
    ):
        return VoiceCommand(action="read_kpi", value="total")

    if any(phrase in normalized for phrase in ("va a", "aller a", "ouvre", "montre", "affiche")):
        tab = _resolve_tab(normalized)
        if tab:
            return VoiceCommand(action="navigate", tab=tab, message=f"Navigation vers {tab}.")

    tab = _resolve_tab(normalized)
    if tab and any(word in normalized for word in ("onglet", "tab", "page", "section")):
        return VoiceCommand(action="navigate", tab=tab, message=f"Navigation vers {tab}.")

    if "compagnie" in normalized or "airline" in normalized:
        tail = _extract_after_keywords(
            normalized,
            ("filtre compagnie", "compagnie", "airline"),
        )
        if tail and airline_codes:
            code = _resolve_airline(tail, airline_codes, airline_labels)
            if code:
                label = (airline_labels or {}).get(code, code)
                return VoiceCommand(
                    action="filter_airline",
                    value=code,
                    message=f"Filtre compagnie : {label}.",
                )

    if "depart" in normalized or "origine" in normalized:
        tail = _extract_after_keywords(
            normalized,
            ("aeroport de depart", "depart", "origine"),
        )
        if tail and origin_airports:
            airport = _match_option(tail, origin_airports)
            if airport:
                return VoiceCommand(
                    action="filter_origin",
                    value=airport,
                    message=f"Filtre depart : {airport}.",
                )

    if "arrivee" in normalized or "destination" in normalized:
        tail = _extract_after_keywords(
            normalized,
            ("aeroport d arrivee", "arrivee", "destination"),
        )
        if tail and destination_airports:
            airport = _match_option(tail, destination_airports)
            if airport:
                return VoiceCommand(
                    action="filter_destination",
                    value=airport,
                    message=f"Filtre arrivee : {airport}.",
                )

    if "mois" in normalized:
        tail = _extract_after_keywords(normalized, ("mois",))
        if tail:
            if tail.isdigit():
                return VoiceCommand(
                    action="filter_month",
                    value=int(tail),
                    message=f"Filtre mois : {tail}.",
                )
            for name, month_id in MONTH_NAMES.items():
                if name in tail:
                    return VoiceCommand(
                        action="filter_month",
                        value=month_id,
                        message=f"Filtre mois : {name.capitalize()}.",
                    )

    return VoiceCommand(
        action="unknown",
        message=(
            f"Commande non reconnue : « {text.strip()} ». "
            "Essayez par ex. « va a compagnies », « filtre compagnie AA », « mois juin »."
        ),
    )


def parse_voice_command(
    text: str,
    airline_codes: list[str] | None = None,
    origin_airports: list[str] | None = None,
    destination_airports: list[str] | None = None,
    airline_labels: dict[str, str] | None = None,
) -> VoiceCommand:
    """Ollama en priorite, regles locales en secours."""
    try:
        return parse_voice_command_ollama(
            text,
            airline_codes=airline_codes,
            origin_airports=origin_airports,
            destination_airports=destination_airports,
            airline_labels=airline_labels,
        )
    except Exception:
        return parse_voice_command_rules(
            text,
            airline_codes=airline_codes,
            origin_airports=origin_airports,
            destination_airports=destination_airports,
            airline_labels=airline_labels,
        )


def process_voice_transcript(
    text: str,
    airline_codes: list[str] | None = None,
    origin_airports: list[str] | None = None,
    destination_airports: list[str] | None = None,
    airline_labels: dict[str, str] | None = None,
) -> VoiceCommand:
    """Applique le mot d'activation puis interprete la commande."""
    command_text, error = extract_command_after_wake_word(text)
    if command_text is None:
        return VoiceCommand(action="unknown", message=error)

    return parse_voice_command(
        command_text,
        airline_codes=airline_codes,
        origin_airports=origin_airports,
        destination_airports=destination_airports,
        airline_labels=airline_labels,
    )


def apply_voice_command(
    command: VoiceCommand,
    kpi_df: pd.DataFrame | None = None,
) -> None:
    """Applique une commande vocale via st.session_state."""
    import streamlit as st

    if command.action == "navigate" and command.tab:
        st.session_state.active_tab = command.tab
        st.session_state.voice_feedback = command.message
        return

    if command.action == "clear_filters":
        st.session_state.filter_airlines = []
        st.session_state.filter_months = []
        st.session_state.filter_origins = []
        st.session_state.filter_destinations = []
        st.session_state.filter_weekdays = []
        st.session_state.filter_periods = []
        st.session_state.voice_feedback = command.message
        return

    if command.action == "filter_airline" and command.value is not None:
        current = list(st.session_state.get("filter_airlines", []))
        value = str(command.value)
        if value not in current:
            current.append(value)
        st.session_state.filter_airlines = current
        st.session_state.voice_feedback = command.message
        return

    if command.action == "filter_origin" and command.value is not None:
        current = list(st.session_state.get("filter_origins", []))
        value = str(command.value)
        if value not in current:
            current.append(value)
        st.session_state.filter_origins = current
        st.session_state.voice_feedback = command.message
        return

    if command.action == "filter_destination" and command.value is not None:
        current = list(st.session_state.get("filter_destinations", []))
        value = str(command.value)
        if value not in current:
            current.append(value)
        st.session_state.filter_destinations = current
        st.session_state.voice_feedback = command.message
        return

    if command.action == "filter_month" and command.value is not None:
        current = list(st.session_state.get("filter_months", []))
        month = int(command.value)
        if month not in current:
            current.append(month)
        st.session_state.filter_months = current
        st.session_state.voice_feedback = command.message
        return

    if command.action == "read_kpi" and kpi_df is not None:
        indicator_map = {
            "delay_rate": "Taux de retard (%)",
            "total": "Nombre total de vols",
        }
        indicator = indicator_map.get(str(command.value), "")
        if indicator:
            row = kpi_df.loc[kpi_df["indicateur"] == indicator, "valeur"]
            if not row.empty:
                value = row.iloc[0]
                st.session_state.voice_feedback = f"{indicator} : {value}"
                return

    st.session_state.voice_feedback = command.message


def generate_data_synthesis(pending: dict, context: dict) -> str:
    """Genere une synthese narrative des donnees via Ollama (apres commande vocale)."""
    import ollama

    if pending.get("command_action") == "unknown":
        return ""

    system = """Tu es un analyste operations aeriennes pour un dashboard de retards de vols US (2015).
Redige une synthese courte (3 a 5 phrases) en francais, style briefing executif.

Regles strictes :
- Utilise UNIQUEMENT les chiffres presents dans le contexte JSON
- N'invente aucune statistique ni cause non fournie
- Mentionne l'onglet actif et les filtres si pertinents
- Compare echantillon filtre vs reseau quand les deux sont disponibles
- Pas de titre, pas de listes a puces, pas de markdown complexe"""

    user = (
        f"Commande vocale : {pending.get('transcript', '')}\n"
        f"Action executee : {pending.get('command_action')} — {pending.get('command_message', '')}\n\n"
        f"Contexte donnees (JSON) :\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=str)}"
    )

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3},
    )
    content = response.message.content
    return content.strip() if content else "Synthese vide : reponse Ollama sans contenu."


def init_voice_session_state() -> None:
    import streamlit as st

    defaults = {
        "active_tab": TAB_LABELS[0],
        "voice_feedback": "",
        "voice_synthesis": "",
        "voice_pending_synthesis": None,
        "voice_live_transcript": "",
        "voice_live_input_display": "",
        "voice_status_message": "",
        "voice_listen_status": "wake",
        "voice_pending_command": None,
        "last_audio_hash": None,
        "last_transcript": "",
        "filter_airlines": [],
        "filter_months": [],
        "filter_origins": [],
        "filter_destinations": [],
        "filter_weekdays": [],
        "filter_periods": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
