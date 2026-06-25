import os
from pathlib import Path

from faster_whisper import WhisperModel


class LocalSTTEngine:
    """
    Local speech-to-text engine using faster-whisper.

    It keeps Whisper local while using better transcription parameters
    for short French dashboard voice commands.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.model_size = model_size

        self.device = device or os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: str = "fr"
    ) -> str:
        """
        Transcribes an audio file into text.

        Optimized for:
        - French voice commands
        - dashboard navigation
        - chatbot activation
        - short spoken phrases
        """

        audio_path = str(audio_path)

        initial_prompt = (
            "Transcription en français de commandes vocales pour un dashboard. "
            "Les mots importants peuvent être : Ok Jack, Jack, Jacques, chatbot, "
            "chatbot désactivé, ferme chatbot, ventes, région, régions, clients, "
            "résumé, tableau de bord, chiffre d'affaires, descends, monte, "
            "réinitialise les filtres. "
            "L'utilisateur peut demander d'afficher les ventes par région, "
            "d'ouvrir le chatbot ou de naviguer dans le dashboard."
        )

        segments, _ = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",

            # Better decoding quality.
            beam_size=5,
            best_of=5,
            temperature=0.0,

            # Important for short commands.
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,

            # We already handle silence in the listener,
            # so avoid Whisper VAD cutting the beginning.
            vad_filter=False,

            # Less aggressive no-speech filtering.
            no_speech_threshold=0.25,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,

            word_timestamps=False,
        )

        transcription_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcription_parts.append(text)

        transcription = " ".join(transcription_parts)
        transcription = self.clean_transcription(transcription)

        return transcription

    def clean_transcription(self, transcription: str) -> str:
        """
        Cleans Whisper output lightly without destroying useful content.
        """

        if not transcription:
            return ""

        transcription = transcription.strip()

        replacements = {
            "Ok, Jack": "Ok Jack",
            "OK Jack": "Ok Jack",
            "Ok Jacques": "Ok Jacques",
            "Jacque": "Jacques",
            "chat bot": "chatbot",
            "tchat bot": "chatbot",
        }

        for old_value, new_value in replacements.items():
            transcription = transcription.replace(old_value, new_value)

        return transcription