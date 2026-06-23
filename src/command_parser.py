import unicodedata
from difflib import SequenceMatcher


class CommandParser:
    """
    Converts transcribed text into dashboard commands.

    This layer is important because STT is not always perfect.
    Example:
    - "zente" can be interpreted as "ventes"
    - "évente" can be interpreted as "ventes"
    """

    def __init__(self):
        self.pages = {
            "ventes": [
                "vente",
                "ventes",
                "zente",
                "zentes",
                "evente",
                "evenement"
            ],
            "clients": [
                "client",
                "clients",
                "utilisateur",
                "utilisateurs"
            ],
            "regions": [
                "region",
                "regions",
                "région",
                "régions"
            ],
            "resume": [
                "resume",
                "résumé",
                "synthese",
                "synthèse"
            ]
        }

        self.metrics = {
            "chiffre_affaires": [
                "chiffre affaires",
                "chiffre d affaires",
                "ca",
                "revenu",
                "revenus"
            ],
            "ventes": [
                "vente",
                "ventes",
                "zente",
                "zentes",
                "evente",
                "évente"
            ],
            "clients": [
                "client",
                "clients",
                "utilisateurs",
                "utilisateur"
            ],
        }

        self.dimensions = {
            "region": [
                "region",
                "regions",
                "région",
                "régions"
            ],
            "mois": [
                "mois",
                "mensuel",
                "janvier",
                "fevrier",
                "février",
                "mars"
            ],
            "produit": [
                "produit",
                "produits",
                "article",
                "articles"
            ]
        }

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

    def similarity(self, word_a: str, word_b: str) -> float:
        return SequenceMatcher(None, word_a, word_b).ratio()

    def contains_or_similar(
        self,
        text: str,
        aliases: list[str],
        threshold: float = 0.72
    ) -> bool:
        words = text.split()

        for alias in aliases:
            normalized_alias = self.normalize(alias)

            if normalized_alias in text:
                return True

            for word in words:
                if self.similarity(word, normalized_alias) >= threshold:
                    return True

        return False

    def find_match(self, text: str, dictionary: dict) -> str | None:
        for key, aliases in dictionary.items():
            if self.contains_or_similar(text, aliases):
                return key

        return None

    def parse_scroll_command(self, text: str) -> dict | None:
        small_amount = 300
        normal_amount = 650
        large_amount = 1100

        go_top_phrases = [
            "retourne en haut",
            "tout en haut",
            "haut de page",
            "reviens en haut"
        ]

        go_bottom_phrases = [
            "tout en bas",
            "bas de page",
            "va tout en bas"
        ]

        scroll_down_small_phrases = [
            "descends un peu",
            "descend un peu",
            "un peu plus bas",
            "legerement plus bas",
            "légèrement plus bas"
        ]

        scroll_up_small_phrases = [
            "monte un peu",
            "remonte un peu",
            "un peu plus haut",
            "legerement plus haut",
            "légèrement plus haut"
        ]

        scroll_down_large_phrases = [
            "descends beaucoup",
            "descend beaucoup",
            "beaucoup plus bas",
            "descends plus vite",
            "page suivante"
        ]

        scroll_up_large_phrases = [
            "monte beaucoup",
            "remonte beaucoup",
            "beaucoup plus haut",
            "monte plus vite",
            "page precedente",
            "page précédente"
        ]

        scroll_down_normal_phrases = [
            "descends",
            "descend",
            "descendre",
            "scroll down",
            "defile vers le bas",
            "défile vers le bas",
            "va en bas",
            "plus bas"
        ]

        scroll_up_normal_phrases = [
            "monte",
            "remonte",
            "remonter",
            "scroll up",
            "defile vers le haut",
            "défile vers le haut",
            "va en haut",
            "plus haut"
        ]

        for phrase in go_top_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "top",
                    "amount": 0,
                    "raw_text": text
                }

        for phrase in go_bottom_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "bottom",
                    "amount": 0,
                    "raw_text": text
                }

        for phrase in scroll_down_small_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "down",
                    "amount": small_amount,
                    "raw_text": text
                }

        for phrase in scroll_up_small_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "up",
                    "amount": small_amount,
                    "raw_text": text
                }

        for phrase in scroll_down_large_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "down",
                    "amount": large_amount,
                    "raw_text": text
                }

        for phrase in scroll_up_large_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "up",
                    "amount": large_amount,
                    "raw_text": text
                }

        for phrase in scroll_down_normal_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "down",
                    "amount": normal_amount,
                    "raw_text": text
                }

        for phrase in scroll_up_normal_phrases:
            if self.normalize(phrase) in text:
                return {
                    "intent": "scroll",
                    "direction": "up",
                    "amount": normal_amount,
                    "raw_text": text
                }

        return None 

    def parse(self, transcription: str) -> dict:
        text = self.normalize(transcription)

        scroll_command = self.parse_scroll_command(text)

        if scroll_command:
            scroll_command["raw_text"] = transcription
            return scroll_command

        reset_phrases = [
            "reset",
            "reinitialise",
            "reinitialiser",
            "reinitialise les filtres",
            "efface",
            "efface les filtres",
            "supprime les filtres"
        ]

        if any(self.normalize(phrase) in text for phrase in reset_phrases):
            return {
                "intent": "reset_filters",
                "raw_text": transcription
            }

        navigation_phrases = [
            "va",
            "aller",
            "ouvre",
            "affiche la page",
            "page"
        ]

        if any(self.normalize(phrase) in text for phrase in navigation_phrases):
            page = self.find_match(text, self.pages)

            if page:
                return {
                    "intent": "go_to_page",
                    "page": page,
                    "raw_text": transcription
                }

        chart_phrases = [
            "affiche",
            "montre",
            "visualise",
            "voir"
        ]

        if any(self.normalize(phrase) in text for phrase in chart_phrases):
            metric = self.find_match(text, self.metrics)
            dimension = self.find_match(text, self.dimensions)

            return {
                "intent": "show_chart",
                "metric": metric,
                "dimension": dimension,
                "raw_text": transcription
            }

        return {
            "intent": "unknown",
            "raw_text": transcription
        }