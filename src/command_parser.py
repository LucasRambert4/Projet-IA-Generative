import re
import unicodedata


class CommandParser:
    """
    Parses French voice commands into structured dashboard commands.

    Supported intents:
    - go_to_page
    - show_chart
    - reset_filters
    - scroll
    - open_chatbot
    - chatbot_message
    - close_chatbot
    - unknown
    """

    def normalize_text(self, text: str) -> str:
        """
        Normalizes text to make command detection more tolerant.

        Example:
        "Événte par région" -> "evente par region"
        """

        if not text:
            return ""

        text = text.lower().strip()

        text = unicodedata.normalize("NFD", text)
        text = "".join(
            character for character in text
            if unicodedata.category(character) != "Mn"
        )

        text = text.replace("'", " ")
        text = text.replace("-", " ")

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def parse(self, text: str) -> dict:
        """
        Main parser entry point.
        """

        normalized_text = self.normalize_text(text)

        if not normalized_text:
            return {
                "intent": "unknown",
                "raw_text": text
            }

        # Remove wake word if it is still present.
        normalized_text = self.remove_wake_word(normalized_text)

        close_chatbot_command = self.parse_close_chatbot_command(normalized_text, text)
        if close_chatbot_command:
            return close_chatbot_command

        chatbot_command = self.parse_chatbot_command(normalized_text, text)
        if chatbot_command:
            return chatbot_command

        scroll_command = self.parse_scroll_command(normalized_text, text)
        if scroll_command:
            return scroll_command

        reset_command = self.parse_reset_command(normalized_text, text)
        if reset_command:
            return reset_command

        page_command = self.parse_page_command(normalized_text, text)
        if page_command:
            return page_command

        chart_command = self.parse_chart_command(normalized_text, text)
        if chart_command:
            return chart_command

        return {
            "intent": "unknown",
            "raw_text": text
        }

    def remove_wake_word(self, normalized_text: str) -> str:
        """
        Removes wake word variants if the listener did not already remove them.
        """

        wake_variants = [
            "ok jack",
            "okay jack",
            "ok jacques",
            "okay jacques",
            "au cas jack",
            "au cas jacques"
        ]

        cleaned_text = normalized_text

        for variant in wake_variants:
            if cleaned_text.startswith(variant):
                cleaned_text = cleaned_text.replace(variant, "", 1).strip()

        return cleaned_text

    # ==========================================================
    # CHATBOT COMMANDS
    # ==========================================================

    def parse_close_chatbot_command(self, normalized_text: str, raw_text: str) -> dict | None:
        """
        Detects commands used to close the chatbot mode.
        """

        close_keywords = [
                "ferme chatbot",
                "ferme le chatbot",
                "fermer chatbot",
                "fermer le chatbot",
                "quitte chatbot",
                "quitte le chatbot",
                "stop chatbot",
                "arrete chatbot",
                "arrete le chatbot",
                "fin chatbot",
                "termine chatbot",
                "termine le chatbot",

                # nouvelles variantes
                "chatbot desactive",
                "chatbot desactivee",
                "chatbot desactiver",
                "desactive chatbot",
                "desactive le chatbot",
                "desactiver chatbot",
                "desactiver le chatbot",
                "coupe chatbot",
                "coupe le chatbot"
            ] 

        if any(keyword in normalized_text for keyword in close_keywords):
            return {
                "intent": "close_chatbot",
                "raw_text": raw_text
            }

        return None

    def parse_chatbot_command(self, normalized_text: str, raw_text: str) -> dict | None:
        """
        Detects commands used to open the chatbot or send a first chatbot message.

        Examples:
        - "chatbot"
        - "ouvre le chatbot"
        - "chatbot explique les ventes"
        - "parle au chatbot explique les clients"
        """

        chatbot_keywords = [
            "ouvre le chatbot",
            "ouvre chatbot",
            "lance le chatbot",
            "lance chatbot",
            "active le chatbot",
            "active chatbot",
            "parle au chatbot",
            "parler au chatbot",
            "assistant vocal",
            "assistant",
            "chat bot",
            "chatbot"
        ]

        matched_keyword = None

        for keyword in chatbot_keywords:
            if keyword in normalized_text:
                matched_keyword = keyword
                break

        if not matched_keyword:
            return None

        message = normalized_text.replace(matched_keyword, "", 1).strip()
        message = message.strip(" ,.;:")

        filler_words = [
            "stp",
            "s il te plait",
            "s il vous plait",
            "peux tu",
            "peut tu",
            "tu peux"
        ]

        for filler in filler_words:
            if message.startswith(filler):
                message = message.replace(filler, "", 1).strip()

        if message:
            return {
                "intent": "chatbot_message",
                "message": message,
                "raw_text": raw_text
            }

        return {
            "intent": "open_chatbot",
            "raw_text": raw_text
        }

    # ==========================================================
    # SCROLL COMMANDS
    # ==========================================================

    def parse_scroll_command(self, normalized_text: str, raw_text: str) -> dict | None:
        """
        Detects page scroll commands.
        """

        top_keywords = [
            "retourne en haut",
            "remonte tout en haut",
            "tout en haut",
            "va en haut",
            "haut de page"
        ]

        bottom_keywords = [
            "tout en bas",
            "va en bas",
            "descend tout en bas",
            "descends tout en bas",
            "bas de page"
        ]

        down_keywords = [
            "descend",
            "descends",
            "defile vers le bas",
            "scroll vers le bas",
            "scroll en bas",
            "plus bas"
        ]

        up_keywords = [
            "monte",
            "remonte",
            "defile vers le haut",
            "scroll vers le haut",
            "scroll en haut",
            "plus haut"
        ]

        if any(keyword in normalized_text for keyword in top_keywords):
            return {
                "intent": "scroll",
                "direction": "top",
                "amount": 0,
                "raw_text": raw_text
            }

        if any(keyword in normalized_text for keyword in bottom_keywords):
            return {
                "intent": "scroll",
                "direction": "bottom",
                "amount": 0,
                "raw_text": raw_text
            }

        if any(keyword in normalized_text for keyword in down_keywords):
            return {
                "intent": "scroll",
                "direction": "down",
                "amount": self.get_scroll_amount(normalized_text),
                "raw_text": raw_text
            }

        if any(keyword in normalized_text for keyword in up_keywords):
            return {
                "intent": "scroll",
                "direction": "up",
                "amount": self.get_scroll_amount(normalized_text),
                "raw_text": raw_text
            }

        return None

    def get_scroll_amount(self, normalized_text: str) -> int:
        """
        Returns scroll amount depending on the voice command.
        """

        if "un peu" in normalized_text or "doucement" in normalized_text:
            return 350

        if "beaucoup" in normalized_text or "fort" in normalized_text:
            return 1200

        return 700

    # ==========================================================
    # RESET COMMANDS
    # ==========================================================

    def parse_reset_command(self, normalized_text: str, raw_text: str) -> dict | None:
        """
        Detects reset commands.
        """

        reset_keywords = [
            "reinitialise",
            "reinitialiser",
            "reset",
            "remet a zero",
            "remets a zero",
            "enleve les filtres",
            "supprime les filtres",
            "retire les filtres",
            "efface les filtres"
        ]

        if any(keyword in normalized_text for keyword in reset_keywords):
            return {
                "intent": "reset_filters",
                "raw_text": raw_text
            }

        return None

    # ==========================================================
    # PAGE COMMANDS
    # ==========================================================

    def parse_page_command(self, normalized_text: str, raw_text: str) -> dict | None:
        """
        Detects navigation commands.
        """

        navigation_keywords = [
            "va a",
            "vas a",
            "aller a",
            "ouvre",
            "affiche la page",
            "montre la page",
            "page"
        ]

        has_navigation_intent = any(
            keyword in normalized_text
            for keyword in navigation_keywords
        )

        if not has_navigation_intent:
            return None

        page = self.detect_page(normalized_text)

        if page:
            return {
                "intent": "go_to_page",
                "page": page,
                "raw_text": raw_text
            }

        return None

    def detect_page(self, normalized_text: str) -> str | None:
        """
        Detects dashboard page name.
        """

        page_keywords = {
            "resume": [
                "resume",
                "accueil",
                "home",
                "dashboard",
                "tableau de bord",
                "general",
                "vue generale"
            ],
            "ventes": [
                "vente",
                "ventes",
                "zente",
                "zentes",
                "evente",
                "eventes",
                "chiffre affaire",
                "chiffre d affaire",
                "revenu",
                "revenus"
            ],
            "clients": [
                "client",
                "clients",
                "utilisateur",
                "utilisateurs",
                "acheteur",
                "acheteurs"
            ],
            "regions": [
                "region",
                "regions",
                "zone",
                "zones",
                "geographie",
                "geographique"
            ]
        }

        for page, keywords in page_keywords.items():
            if any(keyword in normalized_text for keyword in keywords):
                return page

        return None

    # ==========================================================
    # CHART COMMANDS
    # ==========================================================

    def parse_chart_command(self, normalized_text: str, raw_text: str) -> dict | None:
        """
        Detects chart/display commands.

        Examples:
        - "affiche les ventes par region"
        - "montre les clients par region"
        - "affiche le chiffre d affaire par mois"
        """

        display_keywords = [
            "affiche",
            "montre",
            "visualise",
            "donne",
            "voir",
            "je veux voir",
            "montre moi",
            "affiche moi"
        ]

        has_display_intent = any(
            keyword in normalized_text
            for keyword in display_keywords
        )

        if not has_display_intent:
            return None

        metric = self.detect_metric(normalized_text)
        dimension = self.detect_dimension(normalized_text)

        if metric or dimension:
            return {
                "intent": "show_chart",
                "metric": metric,
                "dimension": dimension,
                "raw_text": raw_text
            }

        return None

    def detect_metric(self, normalized_text: str) -> str | None:
        """
        Detects metric from text.
        """

        metric_keywords = {
            "ventes": [
                "vente",
                "ventes",
                "zente",
                "zentes",
                "evente",
                "eventes",
                "revenu",
                "revenus",
                "chiffre affaire",
                "chiffre d affaire",
                "ca"
            ],
            "clients": [
                "client",
                "clients",
                "utilisateur",
                "utilisateurs",
                "acheteur",
                "acheteurs"
            ]
        }

        for metric, keywords in metric_keywords.items():
            if any(keyword in normalized_text for keyword in keywords):
                return metric

        return None

    def detect_dimension(self, normalized_text: str) -> str | None:
        """
        Detects dimension from text.
        """

        dimension_keywords = {
            "region": [
                "region",
                "regions",
                "zone",
                "zones",
                "pays",
                "ville",
                "villes",
                "geographie",
                "geographique"
            ],
            "mois": [
                "mois",
                "date",
                "temps",
                "periode",
                "annee",
                "jour",
                "jours"
            ],
            "client": [
                "client",
                "clients",
                "utilisateur",
                "utilisateurs",
                "acheteur",
                "acheteurs"
            ],
            "produit": [
                "produit",
                "produits",
                "article",
                "articles",
                "categorie",
                "categories"
            ]
        }

        for dimension, keywords in dimension_keywords.items():
            if any(keyword in normalized_text for keyword in keywords):
                return dimension

        return None