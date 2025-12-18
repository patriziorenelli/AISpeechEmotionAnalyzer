from Pipeline.Testo.test_omgdataset import pred_emo_from_omgdataset
from Preprocessing.Testo.omgdataset_preprocess import preprocess_omgdataset_dataset_single_audio
from Utilities.transcription_manager import TranscriptionManager
from collections import Counter, defaultdict
from Utilities.utils import *
import cv2
from Preprocessing.Video.preprocessing_video import *
import shutil
import math
import json
from Pipeline.Video.EmotionExtractor import *
import os
from Pipeline.Audio.AudioEmotionRecognition import AudioEmotionExtractor



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

def segments_to_time_slots_predictions_only(
    emotion_preds: pd.DataFrame,
    emotion_columns: list[str]
) -> dict[int, dict]:
    """
    Converte i segmenti di predizione in time-slot (1 sec)
    restituendo:
      ts -> distribuzione media delle emozioni
    """

    ts_pred_vectors = defaultdict(list)

    for _, row in emotion_preds.iterrows():
        ts_start = math.floor(row["start"]) + 1
        ts_end = math.floor(row["end"]) + 1

        emotion_vector = row[emotion_columns].values.astype(float)

        for ts in range(ts_start, ts_end + 1):
            ts_pred_vectors[ts].append(emotion_vector)

    # media per time-slot
    ts_emotions = {}

    for ts, vectors in ts_pred_vectors.items():
        stacked = np.stack(vectors, axis=0)
        mean_scores = stacked.mean(axis=0)

        ts_emotions[ts] = {
            emotion_columns[i]: float(mean_scores[i])
            for i in range(len(emotion_columns))
        }

    return ts_emotions

def pad_emotions_to_video_length(
    ts_emotions: dict[int, dict],
    total_ts_video: int
) -> dict[int, dict]:
    """
    Estende le emozioni su tutta la durata del video
    usando forward + backward fill.
    """

    padded = {}
    last = None

    # forward fill
    for ts in range(1, total_ts_video + 1):
        if ts in ts_emotions:
            last = ts_emotions[ts]
        padded[ts] = last

    # backward fill per None iniziali
    next_val = None
    for ts in reversed(range(1, total_ts_video + 1)):
        if padded[ts] is None:
            padded[ts] = next_val
        else:
            next_val = padded[ts]

    return padded

def build_gt_per_ts(
    gt_df: pd.DataFrame,
    total_ts_video: int,
    neutral_label: int = 4
) -> dict[int, int]:
    """
    Costruisce la ground truth per ogni time-slot
    usando SOLO gli intervalli originali.
    I buchi → neutral.
    """

    gt_ts = {ts: neutral_label for ts in range(1, total_ts_video + 1)}

    for _, row in gt_df.iterrows():
        ts_start = math.floor(row["start"]) + 1
        ts_end = math.floor(row["end"]) + 1
        label = int(row["EmotionMaxVote"])

        for ts in range(ts_start, ts_end + 1):
            gt_ts[ts] = label

    return gt_ts

def merge_time_slots(
    ts_emotions: dict[int, dict],
    gt_ts: dict[int, int],
    total_ts_video: int
) -> list[dict]:

    time_slots = []

    for ts in range(1, total_ts_video + 1):
        time_slots.append({
            "ts": ts,
            "ground_truth": gt_ts[ts],
            "emotions": ts_emotions.get(ts, {})
        })

    return time_slots


if __name__ == "__main__":

    # Caricamento config
    config = json.load(open("config.json"))
    utils = Utils(config)
    transcription_manager = TranscriptionManager(utils)

    logger = utils.setup_logger()

    #REPO_ID = "PiantoDiGruppo/Ravdess_AML"
    REPO_ID = "PiantoDiGruppo/OMGEmotion_AML"

    if REPO_ID == "PiantoDiGruppo/Ravdess_AML":
        # inserire qui il codice per ravdess
        pass
    else:
        df_metadata = pd.read_csv(utils.config["Paths"]["omgdataset_metadata_file"])

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
                visual_file_path, "extractedFaceFrames", "info.json" )

            emotionExtractor = EmotionExtractor()

            with open(json_file_path, "r", encoding="utf-8") as file:
                dati = json.load(file)

            frames_path = os.path.join( visual_file_path, "extractedFaceFrames") + "/"

            # -------------------- ESTRAZIONE EMOZIONI AUDIO --------------------
            audio_file_path = os.path.join(audio_path, utils.config["General"]["temp_audio_name"])

            #la mia bellissima classe
            audioExtractor = AudioEmotionExtractor()
            #dovrebbe andare bene
            audio_emotions = audioExtractor.predict_per_time_slot(
                audio_path=audio_file_path,
                total_ts=len(dati["time_slot"]),
                slot_seconds=1.0
            )
            # indicizza per lookup rapido
            audio_emotions_by_ts = {x["time_slot"]: x["emotions"] for x in audio_emotions}


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
                    normalized_data["anger"] = normalized_data.pop("angry")

                    logger.info("Emozione dominante: " + max(normalized_data, key=normalized_data.get))
                    logger.info(normalized_data)

                    face_emotions.append({ "time_slot": slot["ts"],  "emotions": normalized_data})
                else:
                    face_emotions.append({ "time_slot": slot["ts"],  "emotions": {}})


            # Nel json salvo il frame rate e il frame_step usato per il downsampling, sono valori che potrebbero esservi utili per capire come allinare i time_slot
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if REPO_ID == "PiantoDiGruppo/OMGEmotion_AML":
                
                logger.info("---------- ANALISI TESTO " + video_name + " -------------")

                # -------------------- PREPROCESSING TESTO --------------------

                #prendiamo audio di patricio (usa direttamente questo)
                audio_file_path = os.path.join(audio_path, utils.config["General"]["temp_audio_name"])

                #carica whisper e fa preprocessing
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
                
                # Mantengo solo le colonne video_id, start, end, label
                ground_truth_labels = df_metadata[["video_id", "start", "end", "EmotionMaxVote"]][df_metadata["video_id"] == vid_name.replace(".mp4","")]
                
                ts_emotions = segments_to_time_slots_predictions_only(
                    emotion_preds,
                    emotion_columns
                )

                ts_emotions = pad_emotions_to_video_length(
                    ts_emotions,
                    len(dati["time_slot"])
                )

                #non mi interessa
                gt_ts = build_gt_per_ts(
                    ground_truth_labels,
                    len(dati["time_slot"]),
                    neutral_label=4
                )

                #manco questa
                time_slots_text = merge_time_slots(
                    ts_emotions,
                    gt_ts,
                    len(dati["time_slot"])
                )

                logger.info("Time slots testo con scores medi:" + str(time_slots_text))

                time_slots = []

                #alla fine i 3 metodi di rpedizione vengono integrati (vedi che manca audio)
                for fe in face_emotions:
                    text_emotions = next((ts for ts in time_slots_text if ts["ts"] == fe["time_slot"]), None)
                    time_slots.append({
                        "ts": fe["time_slot"],
                        "ground_truth": text_emotions["ground_truth"] if text_emotions is not None else None,
                        "modal": {
                            "video": {
                                "emotions": to_python_float(fe["emotions"])
                            },
                            "audio": {
                                "emotions": to_python_float(audio_emotions_by_ts.get(fe["time_slot"], {}))
                            },
                            "text": {
                                "emotions": to_python_float(text_emotions["emotions"]) if text_emotions is not None else {}
                            }    # da integrare
                        }
                    })
            
            #caso altro dataset
            else:
                time_slots = []

                for fe in face_emotions:
                    time_slots.append({
                        "ts": fe["time_slot"],
                        "modal": {
                            "video": {
                                "emotions": to_python_float(fe["emotions"])
                            },
                            "audio": {
                                "emotions": to_python_float(audio_emotions_by_ts.get(fe["time_slot"], {}))
                            },
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
            #shutil.rmtree(general_path)
            break

            # per fermare l'esecuzione quando facciamo test
            #if vid_name == "1v6f9b2KMRA.mp4":
            #   break