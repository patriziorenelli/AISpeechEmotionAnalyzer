# Pipeline/Audio/AudioEmotionExtractor.py

import numpy as np
import librosa
import torch
from transformers import pipeline


class AudioEmotionExtractor:
    """
    Estrae emozioni dall'audio e produce distribuzioni per time-slot.
    Ora include "speech-only scoring" con VAD semplice basato su RMS energy.
    """

    MODEL_TO_EKMAN = {
        "angry": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "joy",
        "neutral": "neutral",
        "sad": "sadness",
        "surprise": "surprise",
    }

    def __init__(
        self,
        model_id: str = "r-f/wav2vec-english-speech-emotion-recognition",
        target_sr: int = 16000,
        device: int | None = None,
        min_seconds: float = 0.25,
        # --- VAD (speech-only scoring) ---
        vad_enabled: bool = True,
        vad_rms_threshold: float = 0.015,      # soglia RMS (da tarare)
        vad_fallback_to_neutral: bool = True,  # se False -> {}
        neutral_fallback_score: float = 1.0,   # usato solo se fallback_to_neutral=True
        # --- opzionale: gate di confidenza ---
        confidence_threshold: float | None = None,  # es. 0.45; None per disabilitare
    ):
        self.target_sr = target_sr
        self.min_seconds = float(min_seconds)

        self.vad_enabled = bool(vad_enabled)
        self.vad_rms_threshold = float(vad_rms_threshold)
        self.vad_fallback_to_neutral = bool(vad_fallback_to_neutral)
        self.neutral_fallback_score = float(neutral_fallback_score)

        self.confidence_threshold = confidence_threshold if confidence_threshold is None else float(confidence_threshold)

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.clf = pipeline(
            task="audio-classification",
            model=model_id,
            framework="pt",
            device=device,
        )

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        audio = audio.astype(np.float32)
        return audio, sr

    @staticmethod
    def _safe_slice(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
        start = max(0, int(round(start_s * sr)))
        end = min(len(audio), int(round(end_s * sr)))
        if end <= start:
            return np.array([], dtype=np.float32)
        return audio[start:end].astype(np.float32)

    def _pad_if_too_short(self, segment: np.ndarray, sr: int) -> np.ndarray:
        min_len = int(round(self.min_seconds * sr))
        if segment.size >= min_len:
            return segment
        pad = min_len - segment.size
        return np.pad(segment, (0, pad), mode="constant").astype(np.float32)

    @staticmethod
    def _rms(segment: np.ndarray) -> float:
        """RMS energy su [-1,1] (librosa.load già normalizza tipicamente in quel range)."""
        if segment.size == 0:
            return 0.0
        # evita overflow e rende robusto
        seg = segment.astype(np.float32)
        return float(np.sqrt(np.mean(seg * seg) + 1e-12))

    def _speech_present(self, segment: np.ndarray) -> bool:
        """
        VAD semplice: se RMS sopra soglia => probabile parlato/suono utile.
        """
        if not self.vad_enabled:
            return True
        return self._rms(segment) >= self.vad_rms_threshold

    def _fallback_output(self) -> dict[str, float]:
        if self.vad_fallback_to_neutral:
            return {"neutral": float(self.neutral_fallback_score)}
        return {}

    def predict_segment(self, segment: np.ndarray, sr: int) -> dict[str, float]:
        """
        Ritorna dizionario emotion->score (rimappato al tuo schema).
        Applica VAD (speech-only scoring) prima di chiamare il modello.
        """
        if segment.size == 0:
            return self._fallback_output()

        # --- speech-only gate ---
        if not self._speech_present(segment):
            return self._fallback_output()

        segment = self._pad_if_too_short(segment, sr)

        preds = self.clf({"array": segment, "sampling_rate": sr}, top_k=None)
        raw_scores = {p["label"].lower(): float(p["score"]) for p in preds}

        mapped = {}
        for k, v in raw_scores.items():
            if k in self.MODEL_TO_EKMAN:
                mapped[self.MODEL_TO_EKMAN[k]] = float(v)

        # rinormalizza
        s = sum(mapped.values())
        if s > 0:
            mapped = {k: (v / s) for k, v in mapped.items()}

        # --- opzionale: confidence gating ---
        if self.confidence_threshold is not None and mapped:
            if max(mapped.values()) < self.confidence_threshold:
                return self._fallback_output()

        return mapped

    def predict_per_time_slot(
        self,
        audio_path: str,
        total_ts: int,
        slot_seconds: float = 1.0,
        start_ts: int = 1,
        context_seconds: float | None = None,  # (opzionale) finestra più lunga
        centered: bool = True,                  # (opzionale) centrata vs lookback
    ) -> list[dict]:
        """
        Predizione per time-slot (1..total_ts).
        Con supporto a finestre audio più lunghe.
        Ora con speech-only scoring integrato.
        """
        audio, sr = self.load_audio(audio_path)

        out = []
        # calcolo soglia RMS adattiva (se vuoi)
        if self.vad_enabled and self.vad_rms_threshold <= 0:
            rms_vals = []
            for ts in range(start_ts, total_ts + 1):
                s0 = (ts - 1) * slot_seconds
                s1 = ts * slot_seconds
                seg0 = self._safe_slice(audio, sr, s0, s1)
                r = self._rms(seg0)
                if r > 1e-6:
                    rms_vals.append(r)

            if rms_vals:
                self.vad_rms_threshold = float(np.percentile(rms_vals, 65))

        for ts in range(start_ts, total_ts + 1):
            base_start_s = (ts - 1) * slot_seconds
            base_end_s = ts * slot_seconds

            if context_seconds is None or context_seconds <= slot_seconds:
                start_s, end_s = base_start_s, base_end_s
            else:
                if centered:
                    center = (base_start_s + base_end_s) / 2.0
                    half = context_seconds / 2.0
                    start_s, end_s = center - half, center + half
                else:
                    start_s, end_s = base_end_s - context_seconds, base_end_s

            seg = self._safe_slice(audio, sr, start_s, end_s)
            scores = self.predict_segment(seg, sr)

            out.append({"time_slot": ts, "emotions": scores})

        return out
