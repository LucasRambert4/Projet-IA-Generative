import re
import unicodedata


DASHBOARD_FACTS = {
    "total_sales": "105 000 €",
    "average_monthly_sales": "17 500 €",
    "best_region": "Nord",
    "worst_region": "Est",
    "worst_region_sales": "19 000 €",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character for character in text
        if unicodedata.category(character) != "Mn"
    )
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_grounded_dashboard_context() -> str:
    return """
Données exactes du dashboard de démonstration :

Indicateurs globaux :
- Ventes totales : 105 000 €
- Vente moyenne mensuelle : 17 500 €

Régions :
- Région avec le plus de ventes : Nord
- Région avec le moins de ventes : Est
- Ventes de la région Est : 19 000 €

Règles obligatoires :
- Répondre uniquement avec les données présentes dans ce contexte.
- Ne jamais inventer une donnée absente.
- Ne jamais associer 19 000 € à la région Nord.
- Si la question demande la meilleure région, répondre Nord.
- Si la question demande la pire région, répondre Est avec 19 000 €.
- Si une information n’existe pas dans le dashboard, dire clairement que le dashboard ne contient pas cette information.
- Le dashboard ne contient pas les ventes de 2029.
- Le dashboard ne contient pas d’adresses IP.
- Le dashboard ne contient pas de salaires.
""".strip()


def answer_from_dashboard_facts(question: str) -> str | None:
    """
    Grounding layer.

    It answers critical dashboard questions from deterministic facts before using the LLM.
    This prevents hallucinations and wrong KPI values.
    """

    normalized_question = normalize_text(question)

    if not normalized_question:
        return None

    asks_to_invent = (
        "invente" in normalized_question
        or "ignore les donnees" in normalized_question
        or "ignore le dashboard" in normalized_question
        or "au hasard" in normalized_question
    )

    if asks_to_invent:
        return (
            "Je ne peux pas inventer une réponse hors contexte. "
            "Je dois répondre uniquement avec les données disponibles dans le dashboard."
        )

    asks_2029 = (
        "2029" in normalized_question
        or "futur" in normalized_question
        or "projection" in normalized_question
    )

    if asks_2029:
        return (
            "Le dashboard ne contient pas les ventes de 2029. "
            "Je ne peux donc pas donner cette information sans l’inventer."
        )

    asks_ip = (
        "adresse ip" in normalized_question
        or "ip des clients" in normalized_question
        or "ip client" in normalized_question
    )

    if asks_ip:
        return (
            "Le dashboard ne contient pas les adresses IP des clients. "
            "Je ne peux donc pas fournir cette information."
        )

    asks_salary = (
        "salaire" in normalized_question
        or "directeur" in normalized_question
        or "employe" in normalized_question
        or "employes" in normalized_question
    )

    if asks_salary:
        return (
            "Le dashboard ne contient pas d’information sur les salaires. "
            "Je ne peux donc pas répondre à cette question sans inventer."
        )

    asks_average_sales = (
        "vente moyenne" in normalized_question
        or "ventes moyenne" in normalized_question
        or "moyenne des ventes" in normalized_question
        or "moyenne de vente" in normalized_question
        or "moyennene" in normalized_question
    )

    if asks_average_sales:
        return (
            "La vente moyenne mensuelle est de 17 500 €. "
            "Elle est calculée à partir des ventes du dashboard."
        )

    asks_total_sales = (
        "total des ventes" in normalized_question
        or "ventes au total" in normalized_question
        or "vente total" in normalized_question
        or "ventes totales" in normalized_question
        or "combien on a fait de ventes" in normalized_question
    )

    if asks_total_sales:
        return "Les ventes totales du dashboard sont de 105 000 €."

    asks_best_region = (
        "region vend le plus" in normalized_question
        or "region marche le mieux" in normalized_question
        or "zone marche le mieux" in normalized_question
        or "meilleure region" in normalized_question
        or "plus performante" in normalized_question
    )

    if asks_best_region:
        return "La région Nord vend le plus dans le dashboard."

    asks_worst_region = (
        "pire region" in normalized_question
        or "region vend le moins" in normalized_question
        or "moins bonne region" in normalized_question
        or "region la plus faible" in normalized_question
    )

    if asks_worst_region:
        return "La pire région est l’Est, avec 19 000 € de ventes."

    return None