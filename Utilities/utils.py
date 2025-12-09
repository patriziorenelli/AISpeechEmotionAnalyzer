import json
import logging
import sys
import shutil


from huggingface_hub import list_repo_files, hf_hub_download
import evaluate
import numpy as np
from sklearn.calibration import LabelEncoder
from transformers import EvalPrediction, PreTrainedTokenizerBase, AutoModelForSequenceClassification, XLNetForSequenceClassification
import pandas as pd

class Utils:
    accuracy: evaluate.Metric
    f1: evaluate.Metric
    precision: evaluate.Metric
    recall: evaluate.Metric
    config: dict

    def __init__(self, config: dict):
        self.config = config
        self.accuracy = evaluate.load(self.config["Metrics"]["accuracy"])
        self.f1 = evaluate.load(self.config["Metrics"]["f1"])
        self.precision = evaluate.load(self.config["Metrics"]["precision"])
        self.recall = evaluate.load(self.config["Metrics"]["recall"])

    def build_label_encoder(self, df_train: pd.DataFrame) -> LabelEncoder:
        le = LabelEncoder()
        logger = self.setup_logger()

        df_train["label"] = le.fit_transform(df_train["label"])

        logger.info("Label mapping: %s", dict(zip(le.classes_, range(len(le.classes_)))))
        
        return le

    def map_labels_to_ekman_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        # Leggo l'ekman mapping dal file ekman_mapping.json
        with open(self.config["Paths"]["ekman_mapping_file"], "r", encoding="utf-8") as f:
            ekman_mapping = dict(json.load(f))
        
        # Mappa le etichette del dataset GoEmotions a quelle di Ekman
        small_to_big = {}
        for big_emo, small_emo in ekman_mapping.items():
            for small_name in small_emo:
                small_to_big[small_name] = big_emo

        # Leggo la lista delle emozione del dataset GoEmotions
        with open(self.config["Paths"]["emotion_file"], "r", encoding="utf-8") as f:
            emotions = [line.strip() for line in f if line.strip()]

        df["label"] = df["label"].apply(lambda i: emotions[i])

        # 5. Mappa nome piccolo in grande emozione
        df["label"] = df["label"].map(small_to_big)

        return df

    def tokenize(self, tokenizer: PreTrainedTokenizerBase , example, max_length:int=512):
        if tokenizer.model_max_length > 10000:
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
        else:
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length)

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = self.accuracy.compute(predictions=predictions, references=labels)
        f1_macro= self.f1.compute(predictions=predictions, references=labels, average="macro")
        f1_micro = self.f1.compute(predictions=predictions, references=labels, average="micro")
        f1_weighted = self.f1.compute(predictions=predictions, references=labels, average="weighted")
        precision_macro = self.precision.compute(predictions=predictions, references=labels, average="macro")
        precision_micro = self.precision.compute(predictions=predictions, references=labels, average="micro")
        precision_weighted = self.precision.compute(predictions=predictions, references=labels, average="weighted")
        recall_macro = self.recall.compute(predictions=predictions, references=labels, average="macro")
        recall_micro = self.recall.compute(predictions=predictions, references=labels, average="micro")
        recall_weighted = self.recall.compute(predictions=predictions, references=labels, average="weighted")

        return {**acc, **f1_macro, **f1_micro, **f1_weighted, **precision_macro, **precision_micro, **precision_weighted, **recall_macro, **recall_micro, **recall_weighted}

    def hp_space(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", self.config["HP_SEARCH"]["min_learning_rate"], self.config["HP_SEARCH"]["max_learning_rate"], log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", json.loads(self.config["HP_SEARCH"]["per_device_train_batch_size"])),
            "num_train_epochs": trial.suggest_int("num_train_epochs", self.config["HP_SEARCH"]["min_num_epochs"], self.config["HP_SEARCH"]["max_num_epochs"]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", self.config["HP_SEARCH"]["min_warmup_ratio"], self.config["HP_SEARCH"]["max_warmup_ratio"]),
            "weight_decay": trial.suggest_float("weight_decay", self.config["HP_SEARCH"]["min_weight_decay"], self.config["HP_SEARCH"]["max_weight_decay"]),
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", json.loads(self.config["HP_SEARCH"]["scheduler_type"])
            )
        }
    
    def setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)

        loglevel = self.config["Logger"]["level"]
        numeric_level = getattr(logging, loglevel.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)

        logger.setLevel(numeric_level)

        # Evita duplicati
        if not logger.handlers:
            formatter = logging.Formatter(
                fmt=self.config["Logger"]["format"],
                datefmt=self.config["Logger"]["date_format"]
            )

            # Console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File (se richiesto)
            log_file_name = self.config["Logger"]["log_file_name"]

            if log_file_name is not None:
                file_handler = logging.FileHandler(log_file_name)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        return logger

    def load_model(self, model_name: str, num_labels: int, model_path:str=None) -> AutoModelForSequenceClassification | XLNetForSequenceClassification:
        if model_path is None:
            model_path = self.config[model_name]["model_name"]

        if model_name == "XLNET":
            try:
                # Proviamo a caricare normalmente
                model = XLNetForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels
                )
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(
                        f"[WARNING] Mismatch nei pesi del modello "
                        f"(checkpoint con un numero diverso di classi). "
                        f"Ricarico con `ignore_mismatched_sizes=True`..."
                    )
                    model = XLNetForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True
                    )
                else:
                    raise e
        elif model_name == "EMOBERTA" or model_name == "DISTILBERT":
            try:
                # Proviamo a caricare normalmente
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels
                )
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(
                        f"[WARNING] Mismatch nei pesi del modello "
                        f"(checkpoint con un numero diverso di classi). "
                        f"Ricarico con `ignore_mismatched_sizes=True`..."
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.config[model_name]["model_name"],
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True
                    )
                else:
                    raise e
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model
    
    def download_single_video_from_hug(repo_id, video_path_in_repo, output_file):
        # Scarica il file
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=video_path_in_repo,
            repo_type="dataset"
        )
        # Copia (o sposta) il file nella destinazione finale
        shutil.copy(local_path, output_file)  # se vuoi rimuovere il file originale, usa shutil.move()

        print(f"Video salvato come {output_file}")


    def get_file_list_names(repo_id):
        # Ottieni la lista di tutti i file nel repo
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        # Filtra quelli nella cartella train
        train_files = [f for f in files if f.startswith("train/")]
        return train_files


        # vediamo numero di frame con viso 
        #  se n > soglia down sampling allora peschiamo gli x con prob più alta
        #  se n = soglia li prendiamo tutti
        #  se n < soglia li prendiamo tutti + frame neri? 
        #  se n = 0 tutti frame neri ?    
    
