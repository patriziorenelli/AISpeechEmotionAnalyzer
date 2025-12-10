import json

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Utilities.utils import Utils
from Preprocessing.Testo.omgdataset_preprocess import load_omgdataset_dataset_file_names, preprocess_omgdataset_dataset_single_video
from Utilities.transcription_manager import TranscriptionManager

def test_omgdataset (model_name: str, dataset_name_train: str, transcription_manager: TranscriptionManager, utils: Utils):
    # Preprocessing
    df_test = None

    logger = utils.setup_logger()

    exclude_neutral = utils.config["Pipelines"]["Text"]["exclude_neutral"]
    n_sample_train = utils.config["Pipelines"]["Text"]["n_sample_train"]
    n_sample_test = utils.config["Pipelines"]["Text"]["n_sample_test"]

    logger.info("=== CARICO MODELLO WHISPER LARGE V3 (PREPROCESSING) ===")
    
    transcriptor, processor, device, torch_dtype = transcription_manager.load_whisper(model_id=utils.config["Preprocessing"]["Text"]["Model"]["model_id"])

    # Preprocessing 
    logger.info(f"Preprocessing del dataset di test OMGDataset")
    video_names_list = load_omgdataset_dataset_file_names(split_name='test', utils=utils, n_samples=n_sample_test, exclude_neutral=exclude_neutral)

    print(video_names_list[0:10])
    print(f"Sono stati caricati {len(video_names_list)} campioni dal dataset OMGDataset test.")

    
    for video_name in video_names_list:
        logger.info(f"\tPreprocessing del video di test: {video_name}")
        preprocess_omgdataset_dataset_single_video(transcription_manager=transcription_manager, utils=utils, device=device, processor=processor,torch_dtype=torch_dtype,transcriptor=transcriptor, video_path=video_name)
        break  # Per ora processo solo un video per testare il funzionamento

    '''
    df_test = preprocess_omgdataset_dataset(split_name='test', n_samples=n_sample_test, exclude_neutral=exclude_neutral, utils=utils)
    '''

config = json.load(open("config.json"))
utils = Utils(config)
transcriptor_manager = TranscriptionManager(utils)

test_omgdataset(
    model_name=config["Pipelines"]["Text"]["Models"]["Emoberta"]["model_name"],
    dataset_name_train="Goemotions",
    transcription_manager=transcriptor_manager,
    utils=utils)