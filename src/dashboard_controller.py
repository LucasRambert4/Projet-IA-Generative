def initialize_dashboard_state(st):
    """
    Initializes Streamlit session state.

    This keeps the dashboard state persistent between interactions.
    """

    default_values = {
        "current_page": "resume",
        "selected_metric": None,
        "selected_dimension": None,
        "last_command": None,
        "last_event_id": None,
        "last_wake_result": None,
        "last_transcription": None,
        "status_message": "Aucune commande exécutée pour le moment."
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_dashboard_command(st, command: dict):
    """
    Applies a parsed voice/text command to the Streamlit dashboard state.
    """

    intent = command.get("intent")
    st.session_state.last_command = command

    if intent == "go_to_page":
        page = command.get("page")

        if page:
            st.session_state.current_page = page
            st.session_state.status_message = f"Navigation vers la page : {page}"
        else:
            st.session_state.status_message = "Page non reconnue."

    elif intent == "show_chart":
        metric = command.get("metric")
        dimension = command.get("dimension")

        st.session_state.selected_metric = metric
        st.session_state.selected_dimension = dimension

        if metric == "ventes":
            st.session_state.current_page = "ventes"
        elif metric == "clients":
            st.session_state.current_page = "clients"

        st.session_state.status_message = (
            f"Affichage demandé : métrique={metric}, dimension={dimension}"
        )

    elif intent == "reset_filters":
        st.session_state.selected_metric = None
        st.session_state.selected_dimension = None
        st.session_state.current_page = "resume"
        st.session_state.status_message = "Filtres réinitialisés."

    else:
        st.session_state.status_message = (
            "Commande non reconnue. Essayez par exemple : "
            "'Va à la page ventes' ou 'Affiche les ventes par région'."
        )


def get_page_label(page_key: str) -> str:
    labels = {
        "resume": "Résumé",
        "ventes": "Ventes",
        "clients": "Clients",
        "regions": "Régions"
    }

    return labels.get(page_key, "Résumé")