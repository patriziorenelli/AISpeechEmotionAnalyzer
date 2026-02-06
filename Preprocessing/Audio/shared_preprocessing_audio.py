import librosa
import numpy as np

def shared_audio_preprocessing(
    audio_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    trim_silence: bool = False,
    top_db: int = 30
):
    """
    Preprocessing condiviso tra Wav2Vec e CNN.
    NON altera la dinamica del segnale.

    Ritorna:
        waveform (np.ndarray), sample_rate
    """

    # Load audio
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=mono)

    # Optional silence trimming (conservativo)
    if trim_silence:
        waveform, _ = librosa.effects.trim(waveform, top_db=top_db)

    # Safety: remove NaN / Inf
    waveform = np.nan_to_num(waveform)

    return waveform, target_sr
