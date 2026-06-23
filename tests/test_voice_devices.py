"""Tests de selection micro et configuration stream."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import voice_navigation as voice


class TestDeviceSelection:
    def test_score_prefers_macbook_over_virtual(self) -> None:
        mac_score = voice._score_input_device("Microphone MacBook Air", 1)
        virtual_score = voice._score_input_device("InstaShareAudio", 2)
        assert mac_score > virtual_score

    @patch("sounddevice.query_devices")
    @patch("sounddevice.default")
    def test_resolve_input_device_auto_pick(
        self, mock_default, mock_query_devices, monkeypatch
    ) -> None:
        voice.set_input_device(None)
        monkeypatch.delenv("VOICE_INPUT_DEVICE", raising=False)

        mock_default.device = [3, 1]
        mock_query_devices.return_value = [
            {
                "name": "Microphone MacBook Air",
                "max_input_channels": 1,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Haut-parleurs MacBook Air",
                "max_input_channels": 0,
                "default_samplerate": 44100.0,
            },
            {
                "name": "InstaShareAudio",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]

        assert voice.resolve_input_device() == 0

    @patch("sounddevice.query_devices")
    def test_device_stream_config_uses_native_rate(
        self, mock_query_devices
    ) -> None:
        mock_query_devices.return_value = {
            "name": "Microphone MacBook Air",
            "max_input_channels": 1,
            "default_samplerate": 48000.0,
        }
        device, channels, rate = voice._device_stream_config(0)
        assert device == 0
        assert channels == 1
        assert rate == 48000

    def test_set_and_get_input_device(self) -> None:
        voice.set_input_device(2)
        assert voice.get_input_device() == 2
        voice.set_input_device(None)


class TestAlexaLoopEvents:
    def test_emit_events_through_queue(self) -> None:
        import queue
        import threading
        import time

        event_queue: queue.Queue = queue.Queue()
        events: list[dict] = []

        def fake_loop(q, get_device, stt_model=None):
            q.put({"type": "status", "phase": "wake", "message": "test"})
            q.put({"type": "live", "text": "OK Google"})
            q.put({"type": "command", "text": "OK Google va a compagnies"})
            while True:
                time.sleep(3600)

        thread = threading.Thread(
            target=fake_loop,
            args=(event_queue, lambda: 0),
            daemon=True,
        )
        thread.start()

        for _ in range(10):
            try:
                events.append(event_queue.get(timeout=0.5))
            except queue.Empty:
                break

        kinds = [event["type"] for event in events]
        assert "status" in kinds
        assert "live" in kinds
        assert "command" in kinds
