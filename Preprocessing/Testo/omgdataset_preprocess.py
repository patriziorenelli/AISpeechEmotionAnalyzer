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

def preprocess_omgdataset_dataset_single_audio(
    transcription_manager: TranscriptionManager, transcriptor: AutoModelForSpeechSeq2Seq, 
    processor: AutoProcessor, 
    device: torch.device, 
    torch_dtype: torch.dtype, 
    utils: Utils, 
    audio_name: str) -> pd.DataFrame:
    logger = utils.setup_logger()

    logger.info("\n=== AVVIO TRASCRIZIONE CON SEGMENTI ===")

    # Faccio il ciclo di trascrizione con chunking
    segments = transcription_manager.transcribe_whisper_with_chunks(
        model=transcriptor,
        processor=processor,
        device=device,
        torch_dtype=torch_dtype,
        audio_path=audio_name,
        language=utils.config["Preprocessing"]["Text"]["Model"]["language"]
    )
 
    preprocessed_dataset = pd.DataFrame(segments)
    print(preprocessed_dataset.head())
    print(f"Number of segments: {len(segments)}")

    return preprocessed_dataset