from Utilities.utils import *
import cv2
from Preprocessing.Video.preprocessing_video import *
import shutil
import json
from Pipeline.Video.EmotionExtractor import *
import os


def to_python_float(obj):
    """
    Converte np.float32, np.float64 ecc. in float Python
    per permettere la serializzazione JSON
    """
    if isinstance(obj, dict):
        return {k: to_python_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_float(v) for v in obj]
    else:
        try:
            return float(obj)
        except:
            return obj


if __name__ == "__main__":

    # Caricamento config
    config = json.load(open("config.json"))
    utils = Utils(config)

    #REPO_ID = "PiantoDiGruppo/Ravdess_AML"
    REPO_ID = "PiantoDiGruppo/OMGEmotion_AML"

    name_list = utils.get_file_list_names(REPO_ID)

    # Crea cartella base
    os.makedirs(config["Paths"]["base_path"], exist_ok=True)

    complete_info_path = config["Paths"]["complete_info_path"]
    if not os.path.exists(complete_info_path):
        with open(complete_info_path, "w", encoding="utf-8") as f:
            json.dump({}, f)


    for video_name in name_list:

        face_emotions = []

        print("---------- ANALISI VIDEO " + video_name + " -------------")

        if "json" not in video_name:

            vid_name = video_name.split("/")[1]

            general_path = os.path.join(config["Paths"]["base_path"], vid_name)
            visual_file_path = os.path.join(general_path, "Video")
            audio_path = os.path.join(general_path, "Audio")

            os.makedirs(visual_file_path, exist_ok=True)
            os.makedirs(audio_path, exist_ok=True)

            # Download video
            utils.download_single_video_from_hug( REPO_ID, video_name, visual_file_path )

            video_path = os.path.join(visual_file_path, vid_name)

            # -------------------- PREPROCESSING VIDEO --------------------
            prep = PreprocessingVideo()
            prep.extract_face_frames_HuggingVersion(video=cv2.VideoCapture(video_path),video_name=vid_name, frame_step=config["Preprocessing"]["Video"]["frame_step"], output_folder=os.path.join(visual_file_path, "extractedFaceFrames"))

            # Estrazione audio
            utils.audioExtraction(video_path, output_path=audio_path)

            # -------------------- ESTRAZIONE EMOZIONI VISO--------------------
            json_file_path = os.path.join(
                visual_file_path, "extractedFaceFrames", "info.json"
            )

            emotionExtractor = EmotionExtractor()

            with open(json_file_path, "r", encoding="utf-8") as file:
                dati = json.load(file)

            frames_path = os.path.join( visual_file_path, "extractedFaceFrames") + "/"

            for slot in dati["time_slot"]:

                if slot["valid"]:
                    print(f"TS: {slot['ts']}")

                    frames_number = len(slot["frames"])
                    #print("Frame analizzati:", frames_number)

                    total = {
                        "angry": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0,
                        "happy": 0.0,
                        "sad": 0.0,
                        "surprise": 0.0,
                        "neutral": 0.0
                    }

                    for frame in slot["frames"]:
                        frame_val = cv2.imread(frames_path + frame)

                        dominant_emotion, all_detected_emotion = ( emotionExtractor.extractFaceEmotion(frame_val))

                        for k, v in all_detected_emotion.items():
                            total[k] += v

                    # Media
                    average = {k: total[k] / frames_number for k in total}

                    # Normalizzazione
                    total_sum = sum(average.values())
                    normalized_data = {k: average[k] / total_sum for k in average}

                    print(  "Emozione dominante:",max(normalized_data, key=normalized_data.get) )
                    print(normalized_data)

                    face_emotions.append({ "time_slot": slot["ts"],  "emotions": normalized_data})

            # Nel json salvo il frame rate e il frame_step usato per il downsampling, sono valori che potrebbero esservi utili per capire come allinare i time_slot
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            cap.release()


            time_slots = []

            for fe in face_emotions:
                time_slots.append({
                    "ts": fe["time_slot"],
                    "modal": {
                        "video": {
                            "emotions": to_python_float(fe["emotions"])
                        },
                        "audio": {},  # da integrare
                        "text": {}    # da integrare
                    }
                })

            # Aggiorno il file di info.json con gli score raccolti
            with open(complete_info_path, "r", encoding="utf-8") as f:
                complete_info = json.load(f)

            complete_info[vid_name] = {
                "nome_file": vid_name,
                "frame_rate": frame_rate,
                "frame_step": config["Preprocessing"]["Video"]["frame_step"],
                "time_slots": time_slots
            }

            with open(complete_info_path, "w", encoding="utf-8") as f:
                json.dump(complete_info, f, indent=4, ensure_ascii=False)

            # cancella tutti i file scaricati e processati, lasciando la cartella principale con dentro solo il file json con le info
            shutil.rmtree(general_path)

            # per fermare l'esecuzione quando facciamo test
            #if vid_name == "1v6f9b2KMRA.mp4":
            #   break

