from collections import deque

import numpy as np


class SileroVADEngine:
    """
    Voice Activity Detection engine using Silero VAD.

    Goal:
    - Detect real human speech instead of relying only on raw microphone volume.
    - Make the listener behave more like a real voice assistant.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        speech_threshold: float = 0.45,
        min_speech_duration_ms: int = 80,
        min_silence_duration_ms: int = 120,
    ):
        self.target_sample_rate = target_sample_rate
        self.speech_threshold = speech_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        torch.set_num_threads(1)

        self.torch = torch
        self.model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps

    def resample_audio(
        self,
        audio: np.ndarray,
        original_sample_rate: int,
    ) -> np.ndarray:
        """
        Resamples audio to 16 kHz for Silero VAD.

        Uses numpy interpolation to avoid adding scipy as another dependency.
        """

        if audio is None:
            return np.array([], dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return audio

        if original_sample_rate == self.target_sample_rate:
            return audio.astype(np.float32)

        duration = audio.size / float(original_sample_rate)
        target_length = int(duration * self.target_sample_rate)

        if target_length <= 0:
            return np.array([], dtype=np.float32)

        original_positions = np.linspace(
            0,
            duration,
            num=audio.size,
            endpoint=False,
        )

        target_positions = np.linspace(
            0,
            duration,
            num=target_length,
            endpoint=False,
        )

        resampled_audio = np.interp(
            target_positions,
            original_positions,
            audio,
        )

        return resampled_audio.astype(np.float32)

    def contains_speech(
        self,
        audio: np.ndarray,
        original_sample_rate: int,
    ) -> tuple[bool, float]:
        """
        Returns whether the given audio window contains speech.

        Also returns RMS energy for debugging.
        """

        if audio is None:
            return False, 0.0

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return False, 0.0

        rms_energy = float(np.sqrt(np.mean(audio ** 2)))

        audio_16k = self.resample_audio(
            audio=audio,
            original_sample_rate=original_sample_rate,
        )

        if audio_16k.size < 512:
            return False, rms_energy

        try:
            tensor = self.torch.from_numpy(audio_16k).float()

            speech_timestamps = self.get_speech_timestamps(
                tensor,
                self.model,
                sampling_rate=self.target_sample_rate,
                threshold=self.speech_threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                return_seconds=False,
            )

            return len(speech_timestamps) > 0, rms_energy

        except Exception:
            return False, rms_energy


class RollingAudioBuffer:
    """
    Keeps the latest audio frames in memory.

    Used for:
    - pre-roll before voice detection
    - VAD rolling window
    """

    def __init__(self, max_frames: int):
        self.frames = deque(maxlen=max_frames)

    def append(self, frame: np.ndarray):
        self.frames.append(frame)

    def clear(self):
        self.frames.clear()

    def to_audio(self) -> np.ndarray:
        if not self.frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(list(self.frames), axis=0).reshape(-1)

    def to_list(self) -> list[np.ndarray]:
        return list(self.frames)