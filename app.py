from datasets import load_dataset
from huggingface_hub import list_repo_files, login, hf_hub_download
import os
import shutil

login("mioTokenHug")
repo_id = "PiantoDiGruppo/AMLDataset2"
video_path_in_repo = "train/01-01-01-01-01-01-01.mp4"
output_file = "D:/Users/Patrizio/Desktop/AISpeeh/AISpeechEmotionAnalyzer/video_completo.mp4"

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


download_single_video_from_hug(repo_id, video_path_in_repo, output_file)