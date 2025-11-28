import os
import torch
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

BERT_MODEL_ID = "dbmdz/bert-base-italian-uncased"   #oppure "bert-base-multilingual-cased"

AUDIO_FILE = r"C:/Users/aless/OneDrive/Desktop/AiSpeech/testvideo.mp4"
LANGUAGE = "it"                     
MODEL_ID = "openai/whisper-large-v3"
TARGET_SR = 16000                   #sample rate



def load_whisper(model_id: str):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] Device: {device}, Dtype: {torch_dtype}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, device, torch_dtype


def load_audio_as_array(audio_path: str, target_sr: int = 16000):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File audio non trovato: {audio_path}")

    print(f"[INFO] Carico audio con librosa: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    print(f"[INFO] Audio shape: {audio.shape}, sample rate: {sr}, durata ~ {len(audio)/sr:.2f}s")
    return audio, sr


#vecchia funzione che non effettua segmentation
def transcribe_whisper(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
    torch_dtype,
    audio_path: str,
    language: str = "it",
):
    audio_array, sr = load_audio_as_array(audio_path, TARGET_SR)

    print("[INFO] Preparo input per il modello...")

    inputs = processor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe",  # oppure "translate"
    )


    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription

#effettua trascrizione con timestamps
def transcribe_whisper_with_chunks(
    model,
    processor,
    device: str,
    torch_dtype,
    audio_path: str,
    language: str = "it",
):
    audio_array, sr = load_audio_as_array(audio_path, TARGET_SR)

    print("[INFO] Preparo input per il modello...")

    inputs = processor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe",
    )

    print("[INFO] Genero trascrizione con timestamps...")

    with torch.no_grad():
        generated = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            return_timestamps=True,          # <-- importantissimo
            return_dict_in_generate=True,
        )

    predicted_ids = generated["sequences"]

    # Qui chiediamo esplicitamente gli offsets
    decoded = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        decode_with_timestamps=False,       # i timestamp li usiamo SOLO per i chunk
        output_offsets=True,                # fa comparire chunks/offsets
    )

    info = decoded[0]

    # Caso 1: versione vecchia di transformers -> è solo una stringa
    if isinstance(info, str):
        print(" La decodifica non restituisce chunks/offsets.")
        print("   Probabile versione vecchia di 'transformers': aggiorna il pacchetto.")
        print("   Testo trascritto:", info[:200], "...")
        return []

    # Caso 2: versione nuova -> cerca prima 'chunks', poi 'offsets'
    chunks = info.get("chunks") or info.get("offsets")

    if not chunks:
        print(" Nessun chunk trovato, struttura info:", info)
        return []

    segments = []
    for ch in chunks:
        start, end = ch["timestamp"]
        segments.append({
            "start": float(start) if start is not None else None,
            "end": float(end) if end is not None else None,
            "text": ch["text"],
        })

    return segments

#Tokenizza i testi dei segmenti con BERT
def tokenize_segments_with_bert(
    segments,
    bert_model_id: str = BERT_MODEL_ID,
    max_length: int = 128,
):
    """
    Ritorna:
      - tokenizer: l'oggetto AutoTokenizer
      - encodings: un dict con input_ids, attention_mask, ecc. (shape: [num_segments, max_length])
    """
    # prendi solo il testo non vuoto
    texts = [s["text"].strip() for s in segments if s["text"].strip()]

    if not texts:
        print("[WARN] Nessun testo valido nei segmenti da tokenizzare.")
        return None, None

    print(f"[INFO] Tokenizzo {len(texts)} segmenti con BERT ({bert_model_id})...")

    #si prende un modello pretrainato di BERT
    tokenizer = AutoTokenizer.from_pretrained(bert_model_id)

    encodings = tokenizer(
        texts,
        padding=True,            # padding a lunghezza massima del batch
        truncation=True,         # tronca se supera max_length
        max_length=max_length,
        return_tensors="pt",     # tensori PyTorch
    )

    print("[INFO] Tokenizzazione completata.")
    print(f"  input_ids shape: {encodings['input_ids'].shape}")
    print(f"  attention_mask shape: {encodings['attention_mask'].shape}")

    return tokenizer, encodings


if __name__ == "__main__":
    print("=== CARICO MODELLO WHISPER LARGE V3 (NO PIPELINE) ===")
    model, processor, device, torch_dtype = load_whisper(MODEL_ID)

    print("\n=== AVVIO TRASCRIZIONE CON SEGMENTI ===")
    segments = transcribe_whisper_with_chunks(
        model=model,
        processor=processor,
        device=device,
        torch_dtype=torch_dtype,
        audio_path=AUDIO_FILE,
        language=LANGUAGE,
)

print("\n=== SEGMENTI ===")
for s in segments:
    print(f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}")

tokenizer, encodings = tokenize_segments_with_bert(segments)

if tokenizer is not None:
    #output tokenizzazione per debug

    first_input_ids = encodings["input_ids"][0]
    print("\n=== ESEMPIO TOKENIZZAZIONE PRIMO SEGMENTO ===")
    for token_id in first_input_ids:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        print(f"  Token ID: {token_id.item():>5}  -->  Token: {token}")

    