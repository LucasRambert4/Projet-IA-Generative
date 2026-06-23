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
            "ventes": ["vente", "ventes", "zente", "zentes", "evente", "evenement"],
            "clients": ["client", "clients", "utilisateur", "utilisateurs"],
            "regions": ["region", "regions", "région", "régions"],
            "resume": ["resume", "résumé", "synthese", "synthèse"]
        }

        self.metrics = {
            "chiffre_affaires": ["chiffre affaires", "chiffre d affaires", "ca", "revenu", "revenus"],
            "ventes": ["vente", "ventes", "zente", "evente"],
            "clients": ["client", "clients", "utilisateurs"],
        }

        self.dimensions = {
            "region": ["region", "regions", "région", "régions"],
            "mois": ["mois", "mensuel", "janvier", "fevrier", "février", "mars"],
            "produit": ["produit", "produits", "article", "articles"]
        }

    def normalize(self, text: str) -> str:
        text = text.lower().strip()

        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        text = text.replace("'", " ")
        text = text.replace("-", " ")
        text = text.replace(".", "")
        text = text.replace(",", "")

        return text

    def similarity(self, word_a: str, word_b: str) -> float:
        return SequenceMatcher(None, word_a, word_b).ratio()

    def contains_or_similar(self, text: str, aliases: list[str], threshold: float = 0.72) -> bool:
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

    def parse(self, transcription: str) -> dict:
        text = self.normalize(transcription)

        if any(word in text for word in ["reset", "reinitialise", "réinitialise", "efface", "supprime les filtres"]):
            return {
                "intent": "reset_filters",
                "raw_text": transcription
            }

        if any(word in text for word in ["va", "aller", "ouvre", "affiche la page", "page"]):
            page = self.find_match(text, self.pages)

            if page:
                return {
                    "intent": "go_to_page",
                    "page": page,
                    "raw_text": transcription
                }

        if any(word in text for word in ["affiche", "montre", "visualise", "voir"]):
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