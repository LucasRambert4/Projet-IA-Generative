"""Tests du mot d'activation et de l'extraction de commande."""

from __future__ import annotations

import pytest

import voice_navigation as voice


class TestWakeWordDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("OK Google va a compagnies", True),
            ("ok gogle va a compagnies", True),
            ("au google compagnies", True),
            ("Okay Google, va a aeroports", True),
            ("ok goggle va a routes", True),
            ("okgoogle routes", True),
            ("bonjour va a compagnies", False),
            ("", False),
        ],
    )
    def test_contains_wake_word(self, text: str, expected: bool) -> None:
        assert voice.contains_wake_word(text) is expected

    def test_extract_command_after_wake_word_success(self) -> None:
        command, error = voice.extract_command_after_wake_word(
            "OK Google, va a compagnies"
        )
        assert error == ""
        assert command == "va a compagnies"

    def test_extract_command_missing_wake_word(self) -> None:
        command, error = voice.extract_command_after_wake_word("va a compagnies")
        assert command is None
        assert "Mot d'activation requis" in error

    def test_extract_command_wake_word_only(self) -> None:
        command, error = voice.extract_command_after_wake_word("OK Google")
        assert command is None
        assert "ajoutez une commande" in error
