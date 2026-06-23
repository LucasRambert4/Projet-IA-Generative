import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from command_parser import CommandParser
from stt_engine import LocalSTTEngine
from audio_utils import save_uploaded_audio_to_temp_file, delete_temp_file
from dashboard_controller import (
    initialize_dashboard_state,
    apply_dashboard_command,
    get_page_label
)


st.set_page_config(
    page_title="Dashboard vocal local",
    page_icon="🎙️",
    layout="wide"
)



@st.cache_data
def load_demo_data():
    data = {
        "mois": ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin"],
        "ventes": [120, 150, 180, 140, 210, 240],
        "clients": [40, 52, 61, 58, 70, 86],
        "region": ["Nord", "Sud", "Est", "Ouest", "Centre", "Nord"]
    }

    return pd.DataFrame(data)


@st.cache_resource
def load_stt_engine():
    """
    Loads the local STT model once.

    Without cache_resource, Streamlit could reload the model too often,
    which would make the app slow.
    """

    return LocalSTTEngine(model_size="base")


def render_sidebar():
    st.sidebar.title("Navigation")

    pages = {
        "resume": "Résumé",
        "ventes": "Ventes",
        "clients": "Clients",
        "regions": "Régions"
    }

    selected_page = st.sidebar.radio(
        "Page active",
        options=list(pages.keys()),
        format_func=lambda page: pages[page],
        index=list(pages.keys()).index(st.session_state.current_page)
        if st.session_state.current_page in pages
        else 0
    )

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page

    st.sidebar.divider()

    st.sidebar.write("État actuel")
    st.sidebar.write(f"Page : `{st.session_state.current_page}`")
    st.sidebar.write(f"Métrique : `{st.session_state.selected_metric}`")
    st.sidebar.write(f"Dimension : `{st.session_state.selected_dimension}`")


def render_voice_command_area():
    st.subheader("Commande vocale")

    st.write(
        "Enregistrez une commande vocale ou utilisez le champ texte pour tester rapidement."
    )

    tab_audio, tab_text = st.tabs(["Micro", "Texte manuel"])

    with tab_audio:
        audio_value = st.audio_input("Enregistrer une commande vocale")

        if audio_value is not None:
            st.audio(audio_value)

        execute_audio = st.button(
            "Transcrire et exécuter",
            use_container_width=True
        )

        if execute_audio:
            if audio_value is None:
                st.warning("Veuillez d'abord enregistrer une commande vocale.")
            else:
                temp_audio_path = save_uploaded_audio_to_temp_file(audio_value)

                try:
                    stt = load_stt_engine()

                    with st.spinner("Transcription locale en cours..."):
                        transcription = stt.transcribe_audio(
                            temp_audio_path,
                            language="fr"
                        )

                    st.session_state.last_transcription = transcription

                    parser = CommandParser()
                    parsed_command = parser.parse(transcription)

                    apply_dashboard_command(st, parsed_command)

                finally:
                    delete_temp_file(temp_audio_path)

    with tab_text:
        command_text = st.text_input(
            "Commande",
            placeholder="Exemple : Affiche les ventes par région"
        )

        execute_text = st.button(
            "Exécuter la commande texte",
            use_container_width=True
        )

        if execute_text:
            parser = CommandParser()
            parsed_command = parser.parse(command_text)
            apply_dashboard_command(st, parsed_command)

    if "last_transcription" in st.session_state:
        st.success(f"Dernière transcription : {st.session_state.last_transcription}")

    st.info(st.session_state.status_message)

    if st.session_state.last_command:
        with st.expander("Voir la commande interprétée"):
            st.json(st.session_state.last_command)

def render_resume_page(df):
    st.header("Résumé du dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total ventes", int(df["ventes"].sum()))

    with col2:
        st.metric("Total clients", int(df["clients"].sum()))

    with col3:
        st.metric("Nombre de régions", int(df["region"].nunique()))

    st.subheader("Données utilisées")
    st.dataframe(df, use_container_width=True)


def render_ventes_page(df):
    st.header("Analyse des ventes")

    if st.session_state.selected_dimension == "region":
        st.subheader("Ventes par région")

        grouped_df = (
            df.groupby("region", as_index=False)["ventes"]
            .sum()
            .sort_values("ventes", ascending=False)
        )

        st.bar_chart(
            grouped_df,
            x="region",
            y="ventes",
            use_container_width=True
        )

        st.dataframe(grouped_df, use_container_width=True)

    else:
        st.subheader("Ventes par mois")

        st.line_chart(
            df,
            x="mois",
            y="ventes",
            use_container_width=True
        )


def render_clients_page(df):
    st.header("Analyse des clients")

    st.subheader("Clients par mois")

    st.bar_chart(
        df,
        x="mois",
        y="clients",
        use_container_width=True
    )


def render_regions_page(df):
    st.header("Analyse par région")

    grouped_df = (
        df.groupby("region", as_index=False)
        .agg(
            ventes=("ventes", "sum"),
            clients=("clients", "sum")
        )
        .sort_values("ventes", ascending=False)
    )

    st.dataframe(grouped_df, use_container_width=True)


def main():
    initialize_dashboard_state(st)

    df = load_demo_data()

    st.title("Dashboard vocal local avec STT")
    st.caption(
        "Prototype local : Whisper/faster-whisper pour le STT, puis interprétation de commandes pour naviguer dans le dashboard."
    )

    render_sidebar()
    render_voice_command_area()

    st.divider()

    current_page = st.session_state.current_page
    page_label = get_page_label(current_page)

    st.write(f"Page affichée : **{page_label}**")

    if current_page == "resume":
        render_resume_page(df)

    elif current_page == "ventes":
        render_ventes_page(df)

    elif current_page == "clients":
        render_clients_page(df)

    elif current_page == "regions":
        render_regions_page(df)

    else:
        render_resume_page(df)


if __name__ == "__main__":
    main()