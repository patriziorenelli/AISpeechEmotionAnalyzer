import json
import pandas as pd

import sys
import os

from transformers import AutoProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Utilities.transcription_manager import TranscriptionManager
from Utilities.utils import *

def load_omgdataset_dataset_file_names(split_name: str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False) -> list[str]:
    logger = utils.setup_logger()

    sample_file_names = utils.get_file_list_names(utils.config["Dataset"]["Omgdataset"]["dataset_name"])

    logger.info(f"Nel dataset OMGDataset {split_name} sono presenti {len(sample_file_names)} campioni.")

    # Elimino tutte le righe corrispondenti all'etichetta "neutral"
    '''
    if exclude_neutral:
        sample_file_names = dataset.filter(lambda x: x["label"] != 27)

    # Seleziono un numero limitato di campioni
    if n_samples is not None and n_samples > 0:
        dataset = dataset.shuffle().select(range(n_samples))

    logger.info(f"Sono stati caricati {len(dataset)} campioni dal dataset RAVDESS {split_name}.")

    df = dataset.to_pandas()
    '''
    return sample_file_names


# load_ravdess_dataset_file_names(split_name='test', utils=utils)

def preprocess_omgdataset_dataset_single_video(
    transcription_manager: TranscriptionManager, transcriptor: AutoModelForSpeechSeq2Seq, 
    processor: AutoProcessor, 
    device: torch.device, 
    torch_dtype: torch.dtype, 
    utils: Utils, 
    video_path: str) -> list[str]:
    logger = utils.setup_logger()
       
    logger.info(f"=== DOWNLOAD VIDEO: {video_path} ===")
    utils.download_single_video_from_hug(utils.config["Dataset"]["Omgdataset"]["dataset_name"], video_path, "tmp_video.mp4")

    logger.info(f"=== ESTRAZIONE AUDIO: {video_path} ===")
    utils.audioExtraction("tmp_video.mp4")

    logger.info("\n=== AVVIO TRASCRIZIONE CON SEGMENTI ===")

    # Faccio il ciclo di trascrizione con chunking
    segments = transcription_manager.transcribe_whisper_with_chunks(
        model=transcriptor,
        processor=processor,
        device=device,
        torch_dtype=torch_dtype,
        audio_path="AudioAnalysis/originalAudio.wav",
        language=utils.config["Preprocessing"]["Text"]["Model"]["language"]
    )
 
    preprocessed_dataset = pd.DataFrame(segments)
    print(preprocessed_dataset.head())
    print("Initial segments:")
    print(segments)
    print(f"Number of segments: {len(segments)}")

    # SONO RIUSCITO AD ARRIVARE ALLA TRASCRIZIONE DEL VIDEO SINGOLO, 
    # ORA DEVO ANALIZZARE OGNI SEGMENTO DEL VIDEO E VEDERE LA LORO ETICHETTA EMOTIVA

    '''
    preprocessed_dataset["video_name"] = ...

    # Carica il dataset da Hugging Face
    df = load_ravdess_dataset(split_name, utils=utils, n_samples=n_samples, exclude_neutral=exclude_neutral)

    # Rimuove testi senza etichette
    df = df[df["labels"].map(lambda x: len(x) > 0)]

    # Prende la prima etichetta e converte a int
    df["label"] = df["labels"].apply(lambda x: int(x[0]))

    # Pulisce testo (qui puoi mettere la tua funzione)
    df["text"] = df["text"].astype(str).str.strip()

    # Mantiene solo colonne necessarie
    df = df[["text", "label"]].reset_index(drop=True)
    '''

    return df