import os
from transformers import AutoTokenizer, TrainingArguments, Trainer
import datasets

from Utilities.utils import Utils
from Preprocessing.Testo.goemotion_preprocessing import preprocess_goemotion_dataset
from Preprocessing.Testo.meld_preprocessing import preprocess_meld_dataset


def test(model_name: str, dataset_name_train: str, dataset_name_test: str, utils: Utils):
    # Preprocessing
    df_test = None

    logger = utils.setup_logger()

    exclude_neutral = utils.config["Pipelines"]["Text"]["exclude_neutral"]
    n_sample_train = utils.config["Pipelines"]["Text"]["n_sample_train"]
    n_sample_test = utils.config["Pipelines"]["Text"]["n_sample_test"]

    # Preprocessing 
    logger.info(f"Preprocessing del dataset di test: {dataset_name_test}")

    if dataset_name_test == "Goemotions":
        df_test = preprocess_goemotion_dataset(split_name='test', n_samples=n_sample_test, exclude_neutral=exclude_neutral, utils=utils)

        # Mappa le etichette a quelle di Ekman
        df_test = utils.map_labels_to_ekman_mapping(df_test)

    elif dataset_name_test == "Meld":
        df_test = preprocess_meld_dataset(split_name='test', n_samples=n_sample_test, exclude_neutral=exclude_neutral, utils=utils)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name_test}")

    logger.info(f"Dataset di test {dataset_name_test} preprocessed. Numero di campioni: {len(df_test)}")
   
    # Codifica le etichette numeriche
    le = utils.build_label_encoder(df_test)

    # Converte direttamente in Dataset Hugging Face
    dataset_test = datasets.Dataset.from_pandas(df_test, preserve_index=False)

    #  Tokenizzazione
    logger.info(f"Tokenizzazione del dataset di test: {dataset_name_test}")

    tokenizer_saved = f"{utils.config['Paths']['tokenizers_dir']}/{model_name.upper()}_{dataset_name_train.upper()}_{n_sample_train}"

    if os.path.exists(tokenizer_saved):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved)
    else:
        tokenizer = AutoTokenizer.from_pretrained(utils.config["Pipelines"]["Text"]["Models"][model_name]["tokenizer_name"])

    tokenized_test = dataset_test.map(lambda x: utils.tokenize(tokenizer, x, utils.config["Pipelines"]["Text"]["Models"][model_name]["tokenizer_max_length"]), batched=True, remove_columns=["text"])

    logger.info(f"Tokenizzazione del dataset di test completata.")

    #  Modello
    logger.info(f"Caricamento del modello {model_name} allenato su {dataset_name_train}")
    
    num_labels = len(le.classes_)
    model_saved = f"{utils.config['Paths']['models_dir']}/{model_name.upper()}_{dataset_name_train.upper()}_{n_sample_train}"

    is_saved = os.path.exists(model_saved)
    model_path = os.path.abspath(model_saved) if is_saved else None

    # Carica il modello
    model = utils.load_model(model_name, num_labels, model_path=model_path)

    #  Trainer (solo valutazione)
    training_args = TrainingArguments(
        output_dir= utils.config["Paths"]["models_dir"],
        fp16=utils.config["Pipelines"]["Text"]["Model"]["fp16"],
        save_strategy="no",
        gradient_accumulation_steps=utils.config["Pipelines"]["Text"]["Model"]["gradient_accumulation_steps"],
        per_device_eval_batch_size=utils.config["Pipelines"]["Text"]["Model"]["per_device_eval_batch_size"],
        dataloader_drop_last=utils.config["Pipelines"]["Text"]["Model"]["dataloader_drop_last"],
        report_to=utils.config["Pipelines"]["Text"]["Model"]["report_to"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        compute_metrics=utils.compute_metrics
    )

    logger.info(f"Modello {model_name} caricato.")

    logger.info(f"Inizio valutazione...")

    # 6. Valutazione
    results = trainer.evaluate()

    logger.info(f"Valutazione completata.")

    logger.info(f"Risultati di {model_name} allenato su {dataset_name_train} e testato su {dataset_name_test}: {results}")