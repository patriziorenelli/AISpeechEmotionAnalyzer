# Pipeline/Audio/AudioEmotionExtractor.py

import numpy as np
import librosa
import torch
from transformers import pipeline


class AudioEmotionExtractor:
    """
    Estrae emozioni dall'audio usando:
    r-f/wav2vec-english-speech-emotion-recognition
    e produce distribuzioni per time-slot da 1 secondo.

    Output label model (tipico): angry, disgust, fear, happy, neutral, sad, surprise
    Rimappate a: anger, disgust, fear, joy, neutral, sadness, surprise
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
    ):
        self.target_sr = target_sr
        self.min_seconds = float(min_seconds)

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

    def predict_segment(self, segment: np.ndarray, sr: int) -> dict[str, float]:
        """
        Ritorna dizionario emotion->score (rimappato al tuo schema).
        Se il segmento è vuoto: {}.
        """
        if segment.size == 0:
            return {}

        segment = self._pad_if_too_short(segment, sr)

        preds = self.clf({"array": segment, "sampling_rate": sr}, top_k=None)

        # normalizza/robustezza: lower-case label e cast float
        raw_scores = {p["label"].lower(): float(p["score"]) for p in preds}

        mapped = {}
        for k, v in raw_scores.items():
            if k in self.MODEL_TO_EKMAN:
                mapped[self.MODEL_TO_EKMAN[k]] = float(v)

        # opzionale: re-normalizzazione (spesso già sommano ~1, ma meglio robusto)
        s = sum(mapped.values())
        if s > 0:
            mapped = {k: (v / s) for k, v in mapped.items()}

        return mapped

    def predict_per_time_slot(
        self,
        audio_path: str,
        total_ts: int,
        slot_seconds: float = 1.0,
        start_ts: int = 1,
    ) -> list[dict]:
        """
        Predizione per time-slot (1..total_ts).
        Ritorna: [{ "time_slot": ts, "emotions": {...} }, ...]
        """
        audio, sr = self.load_audio(audio_path)

        out = []
        for ts in range(start_ts, total_ts + 1):
            start_s = (ts - 1) * slot_seconds
            end_s = ts * slot_seconds

            seg = self._safe_slice(audio, sr, start_s, end_s)
            scores = self.predict_segment(seg, sr)

            out.append({"time_slot": ts, "emotions": scores})

        return out
