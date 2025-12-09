import json
import pandas as pd

from Utilities.utils import Utils

def load_ravdess_dataset_file_names(split_name: str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False):
    logger = utils.setup_logger()

    sample_file_names = utils.get_file_list_names(utils.config["Dataset"]["Ravdess"]["dataset_name"])

    logger.info(f"Nel dataset RAVDESS {split_name} sono presenti {len(sample_file_names)} campioni.")

    print(sample_file_names[0:10])
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






load_ravdess_dataset_file_names(split_name='test', utils=utils)

def preprocess_ravdess_dataset(split_name: str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False) -> pd.DataFrame:
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

    return df