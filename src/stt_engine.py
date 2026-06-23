from faster_whisper import WhisperModel


class LocalSTTEngine:
    """
    Local Speech-To-Text engine using faster-whisper.

    This class is intentionally separated from Streamlit.
    The goal is to keep the STT logic reusable in any interface:
    terminal, Streamlit, Flask, etc.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe_audio(self, audio_path: str, language: str = "fr") -> str:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5
        )

        text_parts = []

        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()