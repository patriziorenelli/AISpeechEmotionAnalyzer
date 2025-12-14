from datasets import load_dataset
from Utilities.utils import Utils
import pandas as pd

def load_goemotion_dataset(split_name: str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False) -> pd.DataFrame:
    logger = utils.setup_logger()

    dataset_path = utils.config["Dataset"]["Goemotions"]["dataset_path"]
    dataset_name = utils.config["Dataset"]["Goemotions"]["dataset_name"]

    dataset = load_dataset(dataset_path, dataset_name, split=split_name)

    logger.info(f"Nel dataset GOMOTIONS {split_name} sono presenti {len(dataset)} campioni.")

    # Elimino tutte le righe corrispondenti all'etichetta "neutral"
    if exclude_neutral:
        dataset = dataset.filter(lambda x: x["label"] != 27)

    # Seleziono un numero limitato di campioni
    if n_samples is not None and n_samples > 0:
        dataset = dataset.shuffle().select(range(n_samples))

    logger.info(f"Sono stati caricati {len(dataset)} campioni dal dataset GOMOTIONS {split_name}.")

    df = dataset.to_pandas()
    return df

def preprocess_goemotion_dataset(split_name: str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False) -> pd.DataFrame:
    # Carica il dataset da Hugging Face
    df = load_goemotion_dataset(split_name, utils=utils, n_samples=n_samples, exclude_neutral=exclude_neutral)

    # Rimuove testi senza etichette
    df = df[df["labels"].map(lambda x: len(x) > 0)]

    # Prende la prima etichetta e converte a int
    df["label"] = df["labels"].apply(lambda x: int(x[0]))

    # Pulisce testo (qui puoi mettere la tua funzione)
    df["text"] = df["text"].astype(str).str.strip()

    # Mantiene solo colonne necessarie
    df = df[["text", "label"]].reset_index(drop=True)

    return df