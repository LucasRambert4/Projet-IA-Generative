"""Tests du pipeline audio PyAV + faster-whisper."""

from __future__ import annotations

import numpy as np
import pytest

import voice_navigation as voice


class TestAudioPipeline:
    def test_resample_48k_to_16k(self) -> None:
        source = np.linspace(0, 1, 48_000, endpoint=False, dtype=np.float32)
        source = (0.2 * np.sin(2 * np.pi * 440 * source)).astype(np.float32)
        resampled = voice.resample_audio_pyav(source, 48_000, 16_000)
        assert resampled.dtype == np.float32
        assert 15_500 <= resampled.size <= 16_500

    def test_decode_wav_roundtrip(self, sample_wav_bytes: bytes) -> None:
        audio, rate = voice.decode_audio_bytes(sample_wav_bytes)
        assert rate == voice.VOICE_SAMPLE_RATE
        assert audio.dtype == np.float32
        assert audio.size == voice.VOICE_SAMPLE_RATE

    def test_prepare_audio_from_numpy(self) -> None:
        audio = np.ones(44_100, dtype=np.float32) * 0.1
        prepared = voice.prepare_audio_for_whisper(audio, 44_100)
        assert prepared.size == 16_000

    def test_prepare_audio_from_bytes(self, sample_wav_bytes: bytes) -> None:
        prepared = voice.prepare_audio_for_whisper(sample_wav_bytes)
        assert prepared.size == voice.VOICE_SAMPLE_RATE

    def test_transcribe_empty_audio(self, mock_whisper_model) -> None:
        empty = np.array([], dtype=np.float32)
        assert voice.transcribe_audio(empty, model=mock_whisper_model) == ""
        mock_whisper_model.transcribe.assert_not_called()

    def test_transcribe_numpy_calls_faster_whisper(
        self, mock_whisper_model
    ) -> None:
        audio = np.ones(voice.VOICE_SAMPLE_RATE, dtype=np.float32) * 0.3
        text = voice.transcribe_audio(audio, model=mock_whisper_model)
        assert text == "OK Google va a compagnies"
        mock_whisper_model.transcribe.assert_called_once()
        passed_audio = mock_whisper_model.transcribe.call_args[0][0]
        assert isinstance(passed_audio, np.ndarray)
        kwargs = mock_whisper_model.transcribe.call_args[1]
        assert kwargs["language"] == "fr"
        assert kwargs["vad_filter"] is True

    def test_transcribe_fast_mode_uses_reduced_beam_size(
        self, mock_whisper_model
    ) -> None:
        audio = np.ones(voice.VOICE_SAMPLE_RATE, dtype=np.float32) * 0.3
        voice.transcribe_audio(audio, model=mock_whisper_model, fast=True)
        assert (
            mock_whisper_model.transcribe.call_args[1]["beam_size"]
            == voice.VOICE_STT_FAST_BEAM_SIZE
        )

    def test_transcribe_chunks_empty(self, mock_whisper_model) -> None:
        assert voice.transcribe_chunks([], 48_000, model=mock_whisper_model) == ""

    def test_transcribe_chunks_with_capture_rate(
        self, mock_whisper_model
    ) -> None:
        chunk = np.ones(1024, dtype=np.float32) * 0.2
        text = voice.transcribe_chunks(
            [chunk, chunk],
            48_000,
            model=mock_whisper_model,
            fast=True,
        )
        assert text == "OK Google va a compagnies"

    def test_threshold_from_chunks(self) -> None:
        loud = np.ones(1024, dtype=np.float32) * 0.5
        quiet = np.ones(1024, dtype=np.float32) * 0.001
        threshold = voice._threshold_from_chunks([quiet, quiet, loud])
        assert threshold >= voice.VOICE_SILENCE_THRESHOLD
