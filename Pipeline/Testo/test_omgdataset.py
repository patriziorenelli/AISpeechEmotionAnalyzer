import os

import datasets
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments

from Utilities.utils import Utils
from Preprocessing.Testo.omgdataset_preprocess import preprocess_omgdataset_dataset_single_audio
from Utilities.transcription_manager import TranscriptionManager
import numpy as np

#Tokenizza i testi dei segmenti con BERT
def tokenize_segments_with_bert(
    segments: pd.DataFrame,
    model_id: str,
    dataset_name_train: str,
    n_sample_train: int,
    utils: Utils,
    max_length: int = 128
) -> pd.DataFrame:
    """
    Ritorna:
    - tokenizer: l'oggetto AutoTokenizer
    - encodings: un dict con input_ids, attention_mask, ecc. (shape: [num_segments, max_length])
    """

    logger = utils.setup_logger()

    if len(segments) == 0:
        logger.warning("Nessun testo valido nei segmenti da tokenizzare.")
        return None, None

    logger.info(f"Tokenizzo {len(segments)} segmenti con ({model_id})...")

    # Converte direttamente in Dataset Hugging Face
    dataset_segments = datasets.Dataset.from_pandas(segments, preserve_index=False)

    #si prende un modello pretrainato di BERT
    tokenizer_saved = f"{utils.config['Paths']['tokenizers_dir']}/{model_id.upper()}_{dataset_name_train.upper()}_{n_sample_train}"

    if os.path.exists(tokenizer_saved):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved)
    else:
        tokenizer = AutoTokenizer.from_pretrained(utils.config["Pipelines"]["Text"]["Models"][model_id]["tokenizer_name"])
    
    tokenized_segments = dataset_segments.map(lambda x: utils.tokenize(tokenizer, x, max_length), batched=True, remove_columns=["text"])

    logger.info("Tokenizzazione completata.")

    return tokenized_segments


def pred_emo_from_omgdataset(
    model_name: str,
    dataset_name_train: str,
    audio_name: str,
    df_segments: pd.DataFrame,
    utils: Utils,
) -> pd.DataFrame:

    df_metadata = pd.read_csv(utils.config["Paths"]["omgdataset_metadata_file"])
    logger = utils.setup_logger()

    n_sample_train = utils.config["Pipelines"]["Text"]["n_sample_train"]

    if len(df_segments) == 0:
        logger.warning("Nessun segmento trovato.")
        return pd.DataFrame()

   
    logger.info(f"=== TOKENIZZAZIONE SEGMENTI ({audio_name}) ===")
    max_length = utils.config["Pipelines"]["Text"]["Models"][model_name]["tokenizer_max_length"]

    dataset_segments = tokenize_segments_with_bert(
        segments=df_segments,
        model_id=model_name,
        dataset_name_train=dataset_name_train,
        n_sample_train=n_sample_train,
        utils=utils,
        max_length=max_length
    )

    if dataset_segments is None or len(dataset_segments) == 0:
        logger.warning("Dataset tokenizzato vuoto.")
        return pd.DataFrame()

    # ===== MODELLO =====
    label_column_name = utils.config["Dataset"]["Omgdataset"]["label_column_name"]
    num_labels = utils.get_num_labels_for_dataset(df_metadata, label_column_name)

    model_saved = f"{utils.config['Paths']['models_dir']}/{model_name.upper()}_{dataset_name_train.upper()}_{n_sample_train}"
    model_path = os.path.abspath(model_saved) if os.path.exists(model_saved) else None

    model = utils.load_model(
        model_name=model_name,
        num_labels=num_labels,
        model_path=model_path
    )

    training_args = TrainingArguments(
        output_dir=utils.config["Paths"]["models_dir"],
        fp16=utils.config["Pipelines"]["Text"]["Models"][model_name]["fp16"],
        per_device_eval_batch_size=utils.config["Pipelines"]["Text"]["Models"][model_name]["per_device_eval_batch_size"],
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args
    )

    logger.info("=== INFERENCE SEGMENT-LEVEL ===")

    predictions = trainer.predict(dataset_segments)

    logits = predictions.predictions

    # Trasformo logits in probabilità con softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


    # Aggiungo nel dataset i risultati per ogni emozione
    id2label = utils.get_emotion_ekman_labels()

    df_out = df_segments.copy()
    for i in range(len(id2label)):
        emotion_label = id2label[i]
        df_out[emotion_label] = probs[:, i]

    pred_ids = np.argmax(logits, axis=1)
    df_out["predicted_label"] = [id2label[int(i)] for i in pred_ids]

    return df_out