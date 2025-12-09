import pandas as pd
from Utilities.utils import Utils

def load_meld_dataset(split_name:str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False) -> pd.DataFrame:
    logger = utils.setup_logger()

    # Usando l'URL possiamo estrarre direttamente solo il testo senza dover scaricare l'intero datase
    base_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"
    dataset = pd.read_csv(base_url + split_name + "_sent_emo.csv")

    logger.info(f"Nel dataset MELD {split_name} sono presenti {len(dataset)} campioni.")

    # Elimino tutte le righe corrispondenti all'etichetta "neutral"
    if exclude_neutral:
        dataset = dataset[dataset["label"] != "neutral"]

    # Seleziono un numero limitato di campioni
    if n_samples is not None and n_samples > 0:
        dataset = dataset.sample(n=n_samples)

    logger.info(f"Sono stati caricati {len(dataset)} campioni dal dataset MELD {split_name}.")

    return dataset

def preprocess_meld_dataset(split_name:str, utils: Utils, n_samples: int=None, exclude_neutral: bool=False) -> pd.DataFrame:
    df = load_meld_dataset(split_name, utils=utils, n_samples=n_samples, exclude_neutral=exclude_neutral)

    # Seleziono solo le colonne necessarie
    df = df[['Dialogue_ID', 'Utterance', 'Emotion']]

    # Rinomino le colonne per la compatibilità
    df = df.rename(columns={'Utterance': 'text', 'Emotion': 'label'}).reset_index(drop=True)

    return df