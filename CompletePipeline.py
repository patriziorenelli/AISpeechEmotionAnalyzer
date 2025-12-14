from Pipeline.Testo.test_omgdataset import pred_emo_from_omgdataset
from Preprocessing.Testo.omgdataset_preprocess import preprocess_omgdataset_dataset_single_audio
from Utilities.transcription_manager import TranscriptionManager
from collections import defaultdict
from Utilities.utils import *
import cv2
from Preprocessing.Video.preprocessing_video import *
import shutil
import math
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

def segments_to_time_slots_with_scores(
    emotion_preds: pd.DataFrame,
    emotion_columns: list
):
    """
    Converte segmenti temporali in time-slot (1 sec)
    restituendo, per ogni time-slot, la distribuzione
    media delle emozioni.

    Esempio di output:
    {
        "ts": 5,
        "emotions": {
            "neutral": 0.19,
            "joy": 0.30,
            "anger": 0.01,
            "sadness": 0.38,
            ...
        }
    }
    """

    # ts -> lista di vettori di probabilità
    ts_emotion_vectors = defaultdict(list)

    for _, row in emotion_preds.iterrows():
        start = row["start"]
        end = row["end"]

        ts_start = math.floor(start) + 1
        ts_end = math.floor(end) + 1

        emotion_vector = row[emotion_columns].values.astype(float)

        for ts in range(ts_start, ts_end + 1):
            ts_emotion_vectors[ts].append(emotion_vector)

    # Aggregazione per time-slot
    time_slot_data = []

    for ts in sorted(ts_emotion_vectors.keys()):
        vectors = np.stack(ts_emotion_vectors[ts], axis=0)
        mean_scores = vectors.mean(axis=0)

        emotions_dict = {
            emotion_columns[i]: float(mean_scores[i])
            for i in range(len(emotion_columns))
        }

        time_slot_data.append({
            "ts": ts,
            "emotions": emotions_dict
        })

    return time_slot_data

def pad_time_slots_to_video_length(
    time_slot_data: list,
    total_ts_video: int
):
    """
    Estende i time-slot audio per coprire tutta la durata del video:
    - backward fill (inizio)
    - forward fill (fine)
    """

    if not time_slot_data:
        raise ValueError("time_slot_data è vuoto")

    # ts -> emotions
    ts_map = {entry["ts"]: entry["emotions"] for entry in time_slot_data}

    first_ts = min(ts_map.keys())
    last_ts = max(ts_map.keys())

    first_emotions = ts_map[first_ts]
    last_emotions = ts_map[last_ts]

    padded_time_slots = []

    for ts in range(1, total_ts_video + 1):
        if ts in ts_map:
            emotions = ts_map[ts]
        elif ts < first_ts:
            # silenzio iniziale → backward fill
            emotions = first_emotions
        else:
            # silenzio finale → forward fill
            emotions = last_emotions

        padded_time_slots.append({
            "ts": ts,
            "emotions": emotions
        })

    return padded_time_slots


if __name__ == "__main__":

    # Caricamento config
    config = json.load(open("config.json"))
    utils = Utils(config)
    transcription_manager = TranscriptionManager(utils)

    logger = utils.setup_logger()

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

        logger.info("---------- ANALISI VIDEO " + video_name + " -------------")

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

                    # Cambio il nome della felicità da happy a joy per uniformità
                    normalized_data["joy"] = normalized_data.pop("happy")

                    logger.info("Emozione dominante: " + max(normalized_data, key=normalized_data.get))
                    logger.info(normalized_data)

                    face_emotions.append({ "time_slot": slot["ts"],  "emotions": normalized_data})

            # Nel json salvo il frame rate e il frame_step usato per il downsampling, sono valori che potrebbero esservi utili per capire come allinare i time_slot
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if REPO_ID == "PiantoDiGruppo/OMGEmotion_AML":
                logger.info("---------- ANALISI TESTO " + video_name + " -------------")

                # -------------------- PREPROCESSING TESTO --------------------

                audio_file_path = os.path.join(audio_path, utils.config["General"]["temp_audio_name"])

                logger.info("=== CARICO MODELLO WHISPER LARGE V3 (PREPROCESSING) ===")
                transcriptor, processor, device, torch_dtype = transcription_manager.load_whisper(
                    model_id=utils.config["Preprocessing"]["Text"]["Model"]["model_id"]
                ) 

                df_segments = preprocess_omgdataset_dataset_single_audio(
                    transcription_manager=transcription_manager,
                    utils=utils,
                    device=device,
                    processor=processor,
                    torch_dtype=torch_dtype,
                    transcriptor=transcriptor,
                    audio_name=audio_file_path
                )

                emotion_preds = pred_emo_from_omgdataset(df_segments=df_segments,model_name="Emoberta", dataset_name_train="Goemotions", utils=utils, audio_name=audio_file_path)

                logger.info(emotion_preds.head())

                emotion_columns = utils.infer_emotion_columns(emotion_preds)
            
                # Converto i segmenti in time_slots con score medi
                time_slots_text = segments_to_time_slots_with_scores(
                    emotion_preds=emotion_preds,
                    emotion_columns=emotion_columns
                )

                # Riempimento time slots per coprire tutta la durata del video
                time_slots_text = pad_time_slots_to_video_length(
                    time_slots_text,
                    len(dati["time_slot"])
                )

                logger.info("Time slots testo con scores medi:" + str(time_slots_text))

                time_slots = []

                for fe in face_emotions:
                    text_emotions = next((ts for ts in time_slots_text if ts["ts"] == fe["time_slot"]), None)
                    time_slots.append({
                        "ts": fe["time_slot"],
                        "modal": {
                            "video": {
                                "emotions": to_python_float(fe["emotions"])
                            },
                            "audio": {},  # da integrare
                            "text": {
                                "emotions": to_python_float(text_emotions["emotions"]) if text_emotions is not None else {}
                            }    # da integrare
                        }
                    })
            else:
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

