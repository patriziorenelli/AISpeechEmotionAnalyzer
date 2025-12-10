import os
import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

from Utilities import utils

class TranscriptionManager:
    def __init__(self, utils: utils.Utils):
       self.utils = utils

    def load_whisper(self,model_id: str):
        logger = self.utils.setup_logger()
    
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Device: {device}, Dtype: {torch_dtype}")

        model = self.utils.load_model(model_id)

        processor = AutoProcessor.from_pretrained(model_id)

        return model, processor, device, torch_dtype


    def load_audio_as_array(self, audio_path: str, target_sr: int = 16000):
        logger = self.utils.setup_logger()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio non trovato: {audio_path}")

        logger.info(f"Carico audio con librosa: {audio_path}")

        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

        logger.info(f"Audio shape: {audio.shape}, sample rate: {sr}, durata ~ {len(audio)/sr:.2f}s")
        return audio, sr


    #vecchia funzione che non effettua segmentation
    def transcribe_whisper(
        self,
        model: AutoModelForSpeechSeq2Seq,
        processor: AutoProcessor,
        device: str,
        torch_dtype,
        audio_path: str,
        language: str = "it",
    ):
        logger = self.utils.setup_logger()

        audio_array, sr = self.load_audio_as_array(audio_path, self.utils.config["Pipelines"]["Text"]["Model"]["target_sr"])

        logger.info("Preparo input per il modello...")

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
        self,
        model,
        processor,
        device: str,
        torch_dtype,
        audio_path: str,
        language: str = "it",
    ):
        logger = self.utils.setup_logger()

        audio_array, sr = self.load_audio_as_array(audio_path, self.utils.config["Pipelines"]["Text"]["Model"]["target_sr"])

        logger.info("Preparo input per il modello...")

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

        logger.info("Genero trascrizione con timestamps...")

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
            logger.warning(" La decodifica non restituisce chunks/offsets.\n Probabile versione vecchia di 'transformers': aggiorna il pacchetto.\n  Testo trascritto: %s ...", info[:200])
            return []

        # Caso 2: versione nuova -> cerca prima 'chunks', poi 'offsets'
        chunks = info.get("chunks") or info.get("offsets")

        if not chunks:
            logger.warning(" Nessun chunk trovato, struttura info:", info)
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
    '''
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
    '''
