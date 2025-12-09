from datasets import load_dataset
from huggingface_hub import list_repo_files, login, hf_hub_download
import shutil
import os


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
    
