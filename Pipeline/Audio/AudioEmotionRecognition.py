from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    import torchaudio.transforms as T
except Exception: 
    torchaudio = None
    T = None

from transformers import AutoModel, AutoProcessor, pipeline


def mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """hidden: [B, T, H], mask: [B, T] bool/int (1 for valid)."""
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.to(hidden.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom



class Wav2Vec2EmbeddingStream(nn.Module):

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base",
        target_sr: int = 16000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.target_sr = int(target_sr)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def forward(self, segment: np.ndarray, sr: int) -> torch.Tensor:
        if sr != self.target_sr:
            segment = librosa.resample(segment.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        inputs = self.processor(segment, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.get("input_values").to(self.device)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        out = self.model(input_values=input_values, attention_mask=attn_mask)
        hidden = out.last_hidden_state  
        emb = mean_pool(hidden, attn_mask)
        return emb.squeeze(0)  


class MelCNN(nn.Module):

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)  
        return self.proj(h) 


class LogMelStream(nn.Module):

    def __init__(
        self,
        target_sr: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        embed_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.target_sr = int(target_sr)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        f_max = float(f_max) if f_max is not None else float(target_sr // 2)

        self.melspec = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            n_mels=int(n_mels),
            f_min=float(f_min),
            f_max=float(f_max),
            power=2.0,
        ).to(self.device)
        self.db = T.AmplitudeToDB(stype="power").to(self.device)
        self.cnn = MelCNN(in_channels=1, embed_dim=int(embed_dim)).to(self.device)
        self.eval()

    @torch.inference_mode()
    def forward(self, segment: np.ndarray, sr: int) -> torch.Tensor:
        if sr != self.target_sr:
            segment = librosa.resample(segment.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        wav = torch.from_numpy(segment.astype(np.float32)).to(self.device)
        wav = wav.unsqueeze(0)  

        mel = self.melspec(wav)  
        mel = self.db(mel)       

        # normalizzazione per segmento (stabile tra clip)
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True).clamp_min(1e-5))

        x = mel.unsqueeze(1) 
        emb = self.cnn(x)     
        return emb.squeeze(0)


class FusionMLP(nn.Module):
    def __init__(self, w2v_dim: int, mel_dim: int, fused_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w2v_dim + mel_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, e_w2v: torch.Tensor, e_mel: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e_w2v, e_mel], dim=-1)
        return self.net(x)


class AudioEmotionExtractor:

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
       
        emotion_model_id: str = "r-f/wav2vec-english-speech-emotion-recognition",
        emotion_scoring_enabled: bool = True,
        wav2vec_embed_model_id: str = "facebook/wav2vec2-base",
        mel_embed_dim: int = 256,
        fused_dim: int = 256,
        target_sr: int = 16000,
        min_seconds: float = 0.25,
        device: int | None = None,
        vad_enabled: bool = True,
        vad_rms_threshold: float = 0.015,
        vad_fallback_to_neutral: bool = True,
        neutral_fallback_score: float = 1.0,
        confidence_threshold: float | None = None,
        mel_n_mels: int = 80,
        mel_n_fft: int = 400,
        mel_hop_length: int = 160,
    ):
        self.target_sr = int(target_sr)
        self.min_seconds = float(min_seconds)

        self.vad_enabled = bool(vad_enabled)
        self.vad_rms_threshold = float(vad_rms_threshold)
        self.vad_fallback_to_neutral = bool(vad_fallback_to_neutral)
        self.neutral_fallback_score = float(neutral_fallback_score)

        self.confidence_threshold = confidence_threshold if confidence_threshold is None else float(confidence_threshold)

        if device is None:
            self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            hf_device = 0 if torch.cuda.is_available() else -1
        else:
            hf_device = int(device)
            if hf_device >= 0:
                self.torch_device = torch.device(f"cuda:{hf_device}")
            else:
                self.torch_device = torch.device("cpu")

        self.emotion_scoring_enabled = bool(emotion_scoring_enabled)
        self.clf = None
        if self.emotion_scoring_enabled:
            self.clf = pipeline(
                task="audio-classification",
                model=emotion_model_id,
                framework="pt",
                device=hf_device,
            )

        self.w2v_stream = Wav2Vec2EmbeddingStream(
            model_id=wav2vec_embed_model_id,
            target_sr=self.target_sr,
            device=self.torch_device,
        )
        self.mel_stream = LogMelStream(
            target_sr=self.target_sr,
            n_mels=mel_n_mels,
            n_fft=mel_n_fft,
            hop_length=mel_hop_length,
            embed_dim=int(mel_embed_dim),
            device=self.torch_device,
        )

        w2v_dim = int(getattr(self.w2v_stream.model.config, "hidden_size", 768))
        self.fusion = FusionMLP(w2v_dim=w2v_dim, mel_dim=int(mel_embed_dim), fused_dim=int(fused_dim)).to(self.torch_device)
        self.fusion.eval()  # senza training, lasciamo eval

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
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
        if segment.size == 0:
            return 0.0
        seg = segment.astype(np.float32)
        return float(np.sqrt(np.mean(seg * seg) + 1e-12))

    def _speech_present(self, segment: np.ndarray) -> bool:
        if not self.vad_enabled:
            return True
        return self._rms(segment) >= self.vad_rms_threshold

    '''
    def _fallback_emotions(self) -> Dict[str, float]:
        if self.vad_fallback_to_neutral:
            return {"neutral": float(self.neutral_fallback_score)}
        return {}
    '''
    def _fallback_emotions(self):
        # fallback uniforme su 7 emozioni EKMAN
        keys = ["neutral","surprise","fear","sadness","joy","anger","disgust"]
        u = 1.0 / len(keys)
        return {k: u for k in keys}


    @torch.inference_mode()
    def extract_fused_embedding(self, segment: np.ndarray, sr: int) -> np.ndarray:
        if segment.size == 0:
            # embedding nullo
            fused_dim = self.fusion.net[-1].out_features if hasattr(self.fusion.net[-1], "out_features") else 256
            return np.zeros((fused_dim,), dtype=np.float32)

        segment = self._pad_if_too_short(segment, sr)

        e_w2v = self.w2v_stream(segment, sr) 
        e_mel = self.mel_stream(segment, sr)  

        e = self.fusion(e_w2v.unsqueeze(0), e_mel.unsqueeze(0)).squeeze(0)
        e = F.normalize(e, dim=-1)
        return e.detach().cpu().to(torch.float32).numpy()

    def predict_emotions_legacy(self, segment: np.ndarray, sr: int) -> Dict[str, float]:

        if not self.emotion_scoring_enabled or self.clf is None:
            return {}

        preds = self.clf({"array": segment, "sampling_rate": sr}, top_k=None)
        raw_scores = {p["label"].lower(): float(p["score"]) for p in preds}

        mapped: Dict[str, float] = {}
        for k, v in raw_scores.items():
            if k in self.MODEL_TO_EKMAN:
                mapped[self.MODEL_TO_EKMAN[k]] = float(v)

        s = sum(mapped.values())
        if s > 0:
            mapped = {k: (v / s) for k, v in mapped.items()}

        if self.confidence_threshold is not None and mapped:
            if max(mapped.values()) < self.confidence_threshold:
                return self._fallback_emotions()

        return mapped

    def predict_segment(
        self,
        segment: np.ndarray,
        sr: int,
        return_embedding: bool = False,
    ) -> Dict[str, Any]:

        if segment.size == 0:
            out: Dict[str, Any] = {"emotions": self._fallback_emotions()}
            if return_embedding:
                out["embedding"] = self.extract_fused_embedding(segment, sr)
            return out

        if not self._speech_present(segment):
            out = {"emotions": self._fallback_emotions()}
            if return_embedding:
                out["embedding"] = self.extract_fused_embedding(segment, sr)
            return out

        segment = self._pad_if_too_short(segment, sr)

        emotions = self.predict_emotions_legacy(segment, sr)
        out2: Dict[str, Any] = {"emotions": emotions}
        if return_embedding:
            out2["embedding"] = self.extract_fused_embedding(segment, sr)
        return out2

    def predict_per_time_slot(
        self,
        audio_path: str,
        total_ts: int,
        slot_seconds: float = 1.0,
        start_ts: int = 1,
        context_seconds: float | None = None,
        centered: bool = True,
        return_embedding: bool = False,
    ) -> List[Dict[str, Any]]:

        audio, sr = self.load_audio(audio_path)

        out: List[Dict[str, Any]] = []

        if self.vad_enabled and self.vad_rms_threshold <= 0:
            rms_vals: List[float] = []
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
            pred = self.predict_segment(seg, sr, return_embedding=return_embedding)

            if return_embedding:
                out.append({"time_slot": ts, "emotions": pred["emotions"], "embedding": pred["embedding"]})
            else:
                out.append({"time_slot": ts, "emotions": pred["emotions"]})

        return out
