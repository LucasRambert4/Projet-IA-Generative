import unicodedata
from difflib import SequenceMatcher


class WakeWordDetector:
    """
    Detects whether a command starts with a wake phrase.

    Example:
    - "Ok Jack, affiche les ventes par région"
    - "Okay Jack affiche les ventes"
    - "Ok Jacques affiche les ventes"

    The goal is to avoid executing accidental speech.
    """

    def __init__(self):
        self.wake_phrases = [
            "ok jack",
            "okay jack",
            "ok jacques",
            "okay jacques",
            "au quai jack",
            "ok jaque",
            "ok jak"
        ]

    def normalize(self, text: str) -> str:
        text = text.lower().strip()

        text = unicodedata.normalize("NFD", text)
        text = "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )

        replacements = {
            "'": " ",
            "-": " ",
            ".": "",
            ",": "",
            "?": "",
            "!": "",
            ":": "",
            ";": ""
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return " ".join(text.split())

    def similarity(self, text_a: str, text_b: str) -> float:
        return SequenceMatcher(None, text_a, text_b).ratio()

    def starts_with_wake_word(self, text: str) -> tuple[bool, str | None]:
        normalized_text = self.normalize(text)

        for wake_phrase in self.wake_phrases:
            normalized_wake_phrase = self.normalize(wake_phrase)

            if normalized_text.startswith(normalized_wake_phrase):
                return True, normalized_wake_phrase

        words = normalized_text.split()

        if len(words) >= 2:
            possible_wake_phrase = " ".join(words[:2])

            for wake_phrase in self.wake_phrases:
                normalized_wake_phrase = self.normalize(wake_phrase)

                if self.similarity(possible_wake_phrase, normalized_wake_phrase) >= 0.75:
                    return True, possible_wake_phrase

        return False, None

    def extract_command(self, text: str) -> dict:
        """
        Returns a structured result:
        - activated: whether the wake word was detected
        - command_text: text after the wake word
        - original_text: original transcription
        """

        normalized_text = self.normalize(text)
        activated, matched_wake_word = self.starts_with_wake_word(text)

        if not activated:
            return {
                "activated": False,
                "wake_word": None,
                "command_text": None,
                "original_text": text
            }

        command_text = normalized_text

        if matched_wake_word and command_text.startswith(matched_wake_word):
            command_text = command_text[len(matched_wake_word):].strip()

        return {
            "activated": True,
            "wake_word": matched_wake_word,
            "command_text": command_text,
            "original_text": text
        }