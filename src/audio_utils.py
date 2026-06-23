from pathlib import Path
from tempfile import NamedTemporaryFile


def save_uploaded_audio_to_temp_file(uploaded_audio) -> str:
    """
    Saves Streamlit microphone audio to a temporary WAV file.

    faster-whisper works well with file paths, so we convert
    the Streamlit audio input into a temporary local file.
    """

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_audio.getvalue())
        return temp_audio_file.name


def delete_temp_file(file_path: str):
    """
    Deletes a temporary file after transcription.
    """

    path = Path(file_path)

    if path.exists():
        path.unlink()