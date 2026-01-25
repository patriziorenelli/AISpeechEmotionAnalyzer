import numpy as np
import librosa
import torch
from transformers import pipeline


class AudioEmotionExtractor:
    """
    Estrae emozioni dall'audio usando:
    r-f/wav2vec-english-speech-emotion-recognition

    Strategia:
    - finestre sliding da 2 secondi
    - stride di 1 secondo
    - la predizione di ogni finestra viene "spalmata"
      sui 2 time-slot da 1 secondo
    - se più finestre contribuiscono allo stesso slot,
      le probabilità vengono mediate

    Output label model:
    angry, disgust, fear, happy, neutral, sad, surprise

    Rimappate a Ekman:
    anger, disgust, fear, joy, neutral, sadness, surprise
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
        min_seconds: float = 2.0,
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

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sr = librosa.load(
            audio_path,
            sr=self.target_sr,
            mono=True,
        )
        return audio.astype(np.float32), sr

    @staticmethod
    def _safe_slice(
        audio: np.ndarray,
        sr: int,
        start_s: float,
        end_s: float,
    ) -> np.ndarray:
        start = max(0, int(round(start_s * sr)))
        end = min(len(audio), int(round(end_s * sr)))

        if end <= start:
            return np.array([], dtype=np.float32)

        return audio[start:end].astype(np.float32)

    def _pad_if_too_short(
        self,
        segment: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        min_len = int(round(self.min_seconds * sr))
        if segment.size >= min_len:
            return segment

        pad = min_len - segment.size
        return np.pad(segment, (0, pad), mode="constant").astype(np.float32)

    # ------------------------------------------------------------------
    # Emotion prediction (clip-based)
    # ------------------------------------------------------------------

    def predict_segment(
        self,
        segment: np.ndarray,
        sr: int,
    ) -> dict[str, float]:
        """
        Predizione clip-based.
        Ritorna: emotion -> probability
        """
        if segment.size == 0:
            return {}

        segment = self._pad_if_too_short(segment, sr)

        preds = self.clf(
            {"array": segment, "sampling_rate": sr},
            top_k=None,
        )

        raw_scores = {
            p["label"].lower(): float(p["score"])
            for p in preds
        }

        mapped = {
            self.MODEL_TO_EKMAN[k]: v
            for k, v in raw_scores.items()
            if k in self.MODEL_TO_EKMAN
        }

        # normalizzazione robusta
        s = sum(mapped.values())
        if s > 0:
            mapped = {k: v / s for k, v in mapped.items()}

        return mapped

    # ------------------------------------------------------------------
    # Sliding window 2s + spalmatura su time-slot 1s
    # ------------------------------------------------------------------

    def predict_per_time_slot(
        self,
        audio_path: str,
        total_ts: int,
        window_seconds: float = 2.0,
        slot_seconds: float = 1.0,
        start_ts: int = 1,
    ) -> list[dict]:
        """
        Finestre sliding da 2s (stride 1s).
        La predizione viene spalmata sui 2 time-slot coperti.
        """

        audio, sr = self.load_audio(audio_path)

        slots_per_window = int(round(window_seconds / slot_seconds))

        # accumulatore: time_slot -> list of emotion dicts
        acc: dict[int, list[dict[str, float]]] = {
            ts: [] for ts in range(start_ts, total_ts + 1)
        }

        max_start_ts = total_ts - slots_per_window + 1

        for ts in range(start_ts, max_start_ts + 1):
            start_s = (ts - 1) * slot_seconds
            end_s = start_s + window_seconds

            segment = self._safe_slice(audio, sr, start_s, end_s)
            scores = self.predict_segment(segment, sr)

            if not scores:
                continue

            # spalma la predizione sui time-slot coperti
            for offset in range(slots_per_window):
                target_ts = ts + offset
                if start_ts <= target_ts <= total_ts:
                    acc[target_ts].append(scores)

        # media finale per ogni time-slot
        out = []
        for ts in range(start_ts, total_ts + 1):
            merged: dict[str, float] = {}

            if acc[ts]:
                for d in acc[ts]:
                    for k, v in d.items():
                        merged[k] = merged.get(k, 0.0) + v

                merged = {
                    k: v / len(acc[ts])
                    for k, v in merged.items()
                }

            out.append(
                {
                    "time_slot": ts,
                    "emotions": merged,
                }
            )

        return out
