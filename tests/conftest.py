"""Fixtures partagees pour les tests vocaux."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

import voice_navigation as voice


class FakeSessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


@pytest.fixture
def streamlit_session(monkeypatch) -> FakeSessionState:
    """Simule st.session_state sans lancer Streamlit."""
    session = FakeSessionState(
        {
            "active_tab": voice.TAB_LABELS[0],
            "voice_feedback": "",
            "filter_airlines": [],
            "filter_months": [],
            "filter_origins": [],
            "filter_destinations": [],
            "filter_weekdays": [],
            "filter_periods": [],
        }
    )
    fake_st = ModuleType("streamlit")
    fake_st.session_state = session
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    return session


@pytest.fixture
def airline_context() -> dict:
    return {
        "airline_codes": ["AA", "DL", "UA"],
        "origin_airports": ["JFK", "LAX", "ORD"],
        "destination_airports": ["SFO", "ATL", "DFW"],
        "airline_labels": {
            "AA": "American Airlines",
            "DL": "Delta Air Lines",
            "UA": "United Airlines",
        },
    }


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """WAV mono 16 kHz genere via le helper du module."""
    audio = np.zeros(voice.VOICE_SAMPLE_RATE, dtype=np.float32)
    audio[:800] = 0.25
    return voice._float_audio_to_wav_bytes(audio, voice.VOICE_SAMPLE_RATE)


@pytest.fixture
def mock_whisper_model():
    """Modele faster-whisper simule."""
    model = MagicMock()

    def _transcribe(audio, **kwargs):
        segment = MagicMock()
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                segment.text = ""
            elif np.max(np.abs(audio)) < 1e-6:
                segment.text = ""
            else:
                segment.text = "OK Google va a compagnies"
        else:
            segment.text = "OK Google va a compagnies"
        return ([segment], {})

    model.transcribe.side_effect = _transcribe
    return model
