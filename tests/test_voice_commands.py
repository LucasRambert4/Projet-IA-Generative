"""Tests du parseur de commandes et de l'application Streamlit."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import voice_navigation as voice


class TestVoiceCommandParsing:
    def test_parse_navigate_compagnies(self) -> None:
        cmd = voice.parse_voice_command_rules("va a compagnies")
        assert cmd.action == "navigate"
        assert cmd.tab == "4. Compagnies"

    def test_parse_clear_filters(self) -> None:
        cmd = voice.parse_voice_command_rules("efface les filtres")
        assert cmd.action == "clear_filters"

    def test_parse_filter_airline_by_code(self, airline_context: dict) -> None:
        cmd = voice.parse_voice_command_rules(
            "filtre compagnie AA",
            airline_codes=airline_context["airline_codes"],
            airline_labels=airline_context["airline_labels"],
        )
        assert cmd.action == "filter_airline"
        assert cmd.value == "AA"

    def test_parse_filter_airline_by_label(self, airline_context: dict) -> None:
        cmd = voice.parse_voice_command_rules(
            "compagnie Delta",
            airline_codes=airline_context["airline_codes"],
            airline_labels=airline_context["airline_labels"],
        )
        assert cmd.action == "filter_airline"
        assert cmd.value == "DL"

    def test_parse_filter_origin(self, airline_context: dict) -> None:
        cmd = voice.parse_voice_command_rules(
            "depart JFK",
            origin_airports=airline_context["origin_airports"],
        )
        assert cmd.action == "filter_origin"
        assert cmd.value == "JFK"

    def test_parse_filter_month_name(self) -> None:
        cmd = voice.parse_voice_command_rules("mois juin")
        assert cmd.action == "filter_month"
        assert cmd.value == 6

    def test_parse_unknown_command(self) -> None:
        cmd = voice.parse_voice_command_rules("fais un cafe")
        assert cmd.action == "unknown"

    def test_process_voice_transcript_full_phrase(
        self, airline_context: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            voice,
            "parse_voice_command_ollama",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        cmd = voice.process_voice_transcript(
            "OK Google, va a compagnies",
            airline_codes=airline_context["airline_codes"],
            origin_airports=airline_context["origin_airports"],
            destination_airports=airline_context["destination_airports"],
            airline_labels=airline_context["airline_labels"],
        )
        assert cmd.action == "navigate"
        assert cmd.tab == "4. Compagnies"

    def test_process_voice_transcript_without_wake_word(self) -> None:
        cmd = voice.process_voice_transcript("va a compagnies")
        assert cmd.action == "unknown"
        assert "Mot d'activation requis" in cmd.message


class TestApplyVoiceCommand:
    def test_apply_navigate(self, streamlit_session) -> None:
        command = voice.VoiceCommand(
            action="navigate",
            tab="4. Compagnies",
            message="Navigation vers 4. Compagnies.",
        )
        voice.apply_voice_command(command)
        assert streamlit_session["active_tab"] == "4. Compagnies"
        assert "Compagnies" in streamlit_session["voice_feedback"]

    def test_apply_filter_airline(self, streamlit_session) -> None:
        command = voice.VoiceCommand(
            action="filter_airline",
            value="AA",
            message="Filtre compagnie : American Airlines.",
        )
        voice.apply_voice_command(command)
        assert streamlit_session["filter_airlines"] == ["AA"]

    def test_apply_clear_filters(self, streamlit_session) -> None:
        streamlit_session["filter_airlines"] = ["AA"]
        streamlit_session["filter_months"] = [6]
        command = voice.VoiceCommand(
            action="clear_filters",
            message="Filtres reinitialises.",
        )
        voice.apply_voice_command(command)
        assert streamlit_session["filter_airlines"] == []
        assert streamlit_session["filter_months"] == []

    def test_apply_read_kpi(self, streamlit_session) -> None:
        kpi_df = pd.DataFrame(
            {
                "indicateur": ["Taux de retard (%)", "Nombre total de vols"],
                "valeur": [18.5, 1_000_000],
            }
        )
        command = voice.VoiceCommand(
            action="read_kpi",
            value="delay_rate",
            message="Lecture KPI.",
        )
        voice.apply_voice_command(command, kpi_df=kpi_df)
        assert "18.5" in streamlit_session["voice_feedback"]


class TestOllamaNormalization:
    def test_normalize_ollama_navigate(self, airline_context: dict) -> None:
        payload = {
            "action": "navigate",
            "tab": "4. Compagnies",
            "value": None,
            "message": "Navigation vers compagnies.",
        }
        cmd = voice._normalize_ollama_command(
            payload,
            airline_context["airline_codes"],
            airline_context["origin_airports"],
            airline_context["destination_airports"],
            airline_context["airline_labels"],
        )
        assert cmd.action == "navigate"
        assert cmd.tab == "4. Compagnies"

    def test_parse_ollama_json_strips_markdown(self) -> None:
        content = '```json\n{"action": "unknown", "message": "test"}\n```'
        parsed = voice._parse_ollama_json(content)
        assert parsed["action"] == "unknown"

    @patch("ollama.Client")
    def test_generate_data_synthesis(self, mock_client_cls) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value.message.content = "Synthese courte des retards."
        pending = {
            "transcript": "OK Google, va a compagnies",
            "command_action": "navigate",
            "command_message": "Navigation vers compagnies.",
            "command_tab": "4. Compagnies",
            "command_value": None,
        }
        context = {"delay_rate": 18.5, "tab": "4. Compagnies"}
        text = voice.generate_data_synthesis(pending, context)
        assert "Synthese courte" in text
        mock_client.chat.assert_called_once()
