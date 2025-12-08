import datasets
from transformers import AutoTokenizer, TrainingArguments, Trainer

from AiSpeechEmotionAnalyzer.logs.LoggerCallback import LoggerCallback
from AiSpeechEmotionAnalyzer.utils import Utils

import os

def train(model_name: str, dataset_name: str, utils: Utils):
    # Preprocessing
    df_train = None
    df_eval = None

    logger = utils.setup_logger()

    exclude_neutral = utils.config["Pipelines"]["Text"]["exclude_neutral"]
    n_sample_train = utils.config["Pipelines"]["Text"]["n_sample_train"]
    n_sample_val = utils.config["Pipelines"]["Text"]["n_sample_val"]

    # Preprocessing 
    logger.info(f"Preprocessing del dataset di addestramento: {dataset_name}")

    if dataset_name == "RAVDESS":
        df_train = preprocess_ravdess_dataset(split_name='train', n_samples=n_sample_train, exclude_neutral=exclude_neutral, utils=utils)
        df_eval = preprocess_ravdess_dataset(split_name='validation', n_samples=n_sample_val, exclude_neutral=exclude_neutral, utils=utils)

        # Mappa le etichette a quelle di Ekman
        df_train = utils.map_labels_to_ekman_mapping(df_train)
        df_eval = utils.map_labels_to_ekman_mapping(df_eval)
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    logger.info(f"Dataset di addestramento {dataset_name} preprocessed. Numero di campioni: {len(df_train)}")

    # Codifica le etichette numeriche
    le = utils.build_label_encoder(df_train)
    df_eval["label"] = le.transform(df_eval["label"])

    # Converte direttamente in Dataset Hugging Face
    dataset_train = datasets.Dataset.from_pandas(df_train, preserve_index=False)
    dataset_eval = datasets.Dataset.from_pandas(df_eval, preserve_index=False)

    #  Tokenizzazione
    logger.info(f"Tokenizzazione del dataset di addestramento: {dataset_name}")

    use_fast = True if model_name != "XLNET" else False  # XLNet non supporta il tokenizer veloce

    tokenizer = AutoTokenizer.from_pretrained(utils.config["Pipelines"]["Text"]["Model"]["tokenizer_name"], use_fast=use_fast)
    tokenized_train = dataset_train.map(lambda x: utils.tokenize(tokenizer, x, utils.config["Pipelines"]["Text"]["Model"]["tokenizer_max_length"]), batched=True, remove_columns=["text"])
    tokenized_eval = dataset_eval.map(lambda x: utils.tokenize(tokenizer, x, utils.config["Pipelines"]["Text"]["Model"]["tokenizer_max_length"]), batched=True, remove_columns=["text"])

    logger.info(f"Tokenizzazione del dataset di addestramento completata.")

    # 7. Modello

    logger.info(f"Caricamento del modello {model_name} allenato su {dataset_name}")

    num_labels = len(le.classes_)
    model = utils.load_model(model_name, num_labels)

    # 9. Argomenti di training
    training_args = TrainingArguments(
        output_dir=utils.config["Paths"]["models_dir"],
        save_strategy=utils.config["Pipelines"]["Text"]["Model"]["save_strategy"],
        fp16=utils.config["Pipelines"]["Text"]["Model"]["fp16"],
        gradient_accumulation_steps=utils.config["Pipelines"]["Text"]["Model"]["gradient_accumulation_steps"],
        eval_strategy="epoch",
        learning_rate=utils.config["Pipelines"]["Text"]["Model"]["learning_rate"],
        per_device_train_batch_size=utils.config["Pipelines"]["Text"]["Model"]["per_device_train_batch_size"],
        per_device_eval_batch_size=utils.config["Pipelines"]["Text"]["Model"]["per_device_eval_batch_size"],
        num_train_epochs=utils.config["Pipelines"]["Text"]["Model"]["num_train_epochs"],
        weight_decay=utils.config["Pipelines"]["Text"]["Model"]["weight_decay"],
        logging_dir= utils.config["Pipelines"]["Text"]["Model"]["logging_dir"],
        logging_steps=utils.config["Pipelines"]["Text"]["Model"]["logging_steps"]
    )
    
    # 10. Trainer
    trainer = Trainer(
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=utils.compute_metrics,
        model_init= lambda: model
    )

    logger.info(f"Trainer per il modello {model_name} creato.")

    trainer.add_callback(LoggerCallback(logger))
    
    do_hpo = utils.config["Pipelines"]["Text"]["Hp_search"]["do_hpo"]

    if do_hpo:
        logger.info("Avvio ricerca iperparametri...")

        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",  # oppure "ray"
            hp_space=utils.hp_space,
            n_trials=utils.config["Pipelines"]["Text"]["Hp_search"]["num_trials"]  # aumenta se hai tempo/GPU
        )
        
        # Applica i migliori iperparametri trovati
        logger.info("Iperparametri migliori trovati:")
        for n, v in best_run.hyperparameters.items():
            logger.info(f" - {n}: {v}")
            setattr(trainer.args, n, v)

        logger.info("Fine ricerca iperparametri.")

    logger.info("Inizio training finale...")

    trainer.train()

    logger.info("Fine training finale.")

    # Creo la directory per salvare il modello e il tokenizer se non esiste
    logger.info(f"Salvataggio del modello e del tokenizer in {model_name}_{dataset_name}_{n_sample_train}...")

    model_save_dir = f"{utils.config['Paths']['models_dir']}/{model_name}_{dataset_name}_{n_sample_train}"
    tokenizer_save_dir = f"{utils.config['Paths']['tokenizers_dir']}/{model_name}_{dataset_name}_{n_sample_train}"

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tokenizer_save_dir, exist_ok=True)

    # Salvo il modello e il tokenizer
    trainer.model.save_pretrained(model_save_dir)
    trainer.tokenizer.save_pretrained(tokenizer_save_dir)

    logger.info(f"Modello e tokenizer salvati in {model_name}_{dataset_name}_{n_sample_train}")

    return trainer, le