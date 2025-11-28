import librosa
import numpy as np
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt
import soundfile as sf

# --- Funzioni di Filtro ---

def highpass_sos(data, cutoff, sr, order=2):
    """Filtro Passa-Alto per rimuovere il rumble (rumore a bassa frequenza)."""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data
    sos = butter(order, normal_cutoff, btype='high', output='sos')
    return sosfiltfilt(sos, data)

def bandpass_sos(data, lowcut, highcut, sr, order=4):
    """Filtro Passa-Banda using SOS for stability."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    
    if high >= 1.0 or low >= high:
        print(f"Warning: Cutoff limits invalid. Returning unfiltered signal.")
        return data

    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def vocal_boost_sos(signal, sr, low_freq, high_freq, gain_db=6):
    """Applica un boost (equalizzazione) in una banda specifica."""
    gain = 10**(gain_db / 20)
    # Applichiamo il filtro passa-banda NON distruttivo solo alla porzione che vogliamo amplificare
    boosted_content = bandpass_sos(signal, low_freq, high_freq, sr)
    # Aggiungiamo il contenuto boostato al segnale originale
    return signal + boosted_content * gain

def normalize(signal, target_dbfs=-0.1):
    """Normalizza il segnale al target dBFS."""
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    # Conversione da dBFS a ampiezza lineare (target < 1.0)
    target_amp = 10**(target_dbfs/20) 
    # Normalizza e poi applica il target_amp
    normalized_signal = signal / max_val
    return normalized_signal * target_amp

# --- CONFIGURAZIONE REVISIONATA ---
INPUT_FILE = "AudioAnalisis/originalAudio.wav"
OUTPUT_FILE = "AudioAnalisis/Audio_cleaned_boosted_revised_v2.wav"

# 1. Filtro Iniziale (rimuove il rumble prima del denoising)
LOW_CUT_HPF = 150 # NUOVO: Taglia tutto ciò che è sotto i 150 Hz (rumble, basse frequenze indesiderate)

# 2. Denoise
DENOISE_STRENGTH = 0.85 # PIÙ AGGRESSIVO (da 0.7)

# 3. Equalizzazione Generale (la banda passante per la voce)
VOCAL_LOWCUT = 100 # Dopo il taglio iniziale, teniamo un filtro morbido a 100 Hz
VOCAL_HIGHCUT = 12000 # Esteso a 12 kHz per preservare più aria/dettaglio
# 4. Boost di Presenza/Intelligibilità
PRESENCE_FREQ_LOW = 2000 # 2.0 kHz (zona di nasali e consonantali chiare)
PRESENCE_FREQ_HIGH = 5000 # 5.0 kHz (zona dell'orecchio più sensibile al parlato)
PRESENCE_GAIN_DB = 7 # PIÙ AGGRESSIVO (da 5 dB)
TARGET_NORMALIZATION_DBFS = -1.0


# --- ESECUZIONE DELLA PIPELINE REVISIONATA ---

# 1. Carica il file audio
try:
    y, sr = librosa.load(INPUT_FILE, sr=None, mono=True)
except FileNotFoundError:
    print(f"ERRORE: File non trovato all'indirizzo: {INPUT_FILE}. Assicurati che il percorso sia corretto.")
    exit()

# 2. Pre-filtro Passa-Alto (Low-Cut) per pulire le basse frequenze
print(f"Step 1/6: Applicazione del filtro Passa-Alto ({LOW_CUT_HPF} Hz) per il Low-Cut...")
y_lowcut = highpass_sos(y, LOW_CUT_HPF, sr)

# 3. Denoise (Riduzione del Rumore)
# prop_decrease più alto (0.85) per maggiore aggressività.
print(f"Step 2/6: Esecuzione del Denoise (Strength={DENOISE_STRENGTH})...")
y_denoised = nr.reduce_noise(y=y_lowcut, sr=sr, prop_decrease=DENOISE_STRENGTH)

# 4. Equalizzazione (Band-pass di pulizia)
# Applicato DOPO il denoising per pulire ulteriormente le estremità dello spettro.
print(f"Step 3/6: Applicazione del filtro Passa-Banda ({VOCAL_LOWCUT} Hz - {VOCAL_HIGHCUT} Hz)...")
y_eq = bandpass_sos(y_denoised, VOCAL_LOWCUT, VOCAL_HIGHCUT, sr)

# 5. Boost di Presenza/Intelligibilità
# Boost mirato alla regione che dà chiarezza e intelligibilità alla voce.
print(f"Step 4/6: Boost di Presenza ({PRESENCE_FREQ_LOW}-{PRESENCE_FREQ_HIGH} Hz, Gain={PRESENCE_GAIN_DB} dB)...")
y_boost = vocal_boost_sos(y_eq, sr, PRESENCE_FREQ_LOW, PRESENCE_FREQ_HIGH, gain_db=PRESENCE_GAIN_DB)

# 6. Normalizzazione
print(f"Step 5/6: Normalizzazione del volume a {TARGET_NORMALIZATION_DBFS} dBFS...")
y_final = normalize(y_boost, target_dbfs=TARGET_NORMALIZATION_DBFS)

# 7. Salva il risultato
sf.write(OUTPUT_FILE, y_final, sr)
print(f"\n✅ Elaborazione completata! File salvato come: {OUTPUT_FILE}")