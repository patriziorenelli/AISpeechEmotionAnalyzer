from Pipeline.FeedbackCoach import FeedbackCoach
from Pipeline.Testo.test_omgdataset import pred_emo_from_omgdataset
from Preprocessing.Testo.omgdataset_preprocess import preprocess_omgdataset_dataset_single_audio
from Utilities.evaluation_manager import EvaluationManager
from Utilities.transcription_manager import TranscriptionManager
from collections import defaultdict
from Utilities.utils import *
import cv2
from Preprocessing.Video.preprocessing_video import *
import shutil
import math
import json
from Pipeline.Video.newEmotionExtractor_ottimale import *
import os
from Pipeline.Audio.AudioEmotionRecognition import AudioEmotionExtractor
import gc
import torch
import os
import random


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

def build_gt_with_intervals(
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

def build_gt_without_intervals(
    total_ts_video: int,
    video_label: int
) -> dict[int, int]:
    """
    Costruisce la ground truth per ogni time-slot
    non usando gli intervalli originali.
    """
    gt_ts = {ts: video_label for ts in range(1, total_ts_video + 1)}

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

def generate_feedback_for_all_videos(complete_info_path: str) -> dict:
    """
    Legge il JSON finale della pipeline e genera feedback testuale
    per ogni video.
    """

    with open(complete_info_path, "r", encoding="utf-8") as f:
        complete_info = json.load(f)

    coach = FeedbackCoach()
    feedback_results = {}

    for video_name, video_data in complete_info.items():
        time_slots = video_data["time_slots"]

        stream_scores = {
            "video": coach.aggregate_stream_scores(time_slots, "video"),
            "audio": coach.aggregate_stream_scores(time_slots, "audio"),
            "text": coach.aggregate_stream_scores(time_slots, "text"),
        }

        feedback_text = coach.generate_full_feedback(stream_scores)

        feedback_results[video_name] = feedback_text

    return feedback_results

if __name__ == "__main__":

    # Caricamento config
    config = json.load(open("config.json"))
    utils = Utils(config)
    transcription_manager = TranscriptionManager(utils)
    evaluation_manager = EvaluationManager(utils)

    logger = utils.setup_logger()
    
    REPO_ID = "PiantoDiGruppo/Ravdess_AML"
    #REPO_ID = "PiantoDiGruppo/OMGEmotion_AML"

    text_model = "Emoberta"
    dataset_train_text = "Goemotions"

    if REPO_ID == "PiantoDiGruppo/Ravdess_AML":
        df_metadata = pd.read_csv(utils.config["Paths"]["ravdess_metadata_file"])
    else:
        df_metadata = pd.read_csv(utils.config["Paths"]["omgdataset_metadata_file"])

    name_list = utils.get_file_list_names(REPO_ID)[0:utils.config["General"]["numVideo"]]

    #-------------------------------------------
    # DA TOGLIERE PER IL TESTING 
    
    """
     for x in name_list:
        print(x)
        print(x.split('-')[2])


    filtrati = [
        s for s in name_list 
        if s.split('-')[2] != '01'
    ]

    # 2. Estraiamo 70 elementi casuali
    # Usiamo min() per evitare errori se i file filtrati sono meno di 70
    n_da_estrarre = min(len(filtrati), 150)
    risultato = random.sample(filtrati, n_da_estrarre)

    print(f"File estratti: {len(risultato)}")
    name_list = risultato
    """


    #------------------------------------------


    # Crea cartella base
    os.makedirs(config["Paths"]["base_path"], exist_ok=True)

    complete_info_path = config["Paths"]["complete_info_path"]
    if not os.path.exists(complete_info_path):
        with open(complete_info_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    logger.info("=== INIZIO PIPELINE COMPLETA ===")

    emotionExtractor = NewEmotionExtractor(model_path="checkpoints/best_model.pt")

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

            #emotionExtractor = NewEmotionExtractor()  # qui da mettere metodo che sfrutta poi newEmotionExtractor.py per estrarre le emozioni dai frame 

            with open(json_file_path, "r", encoding="utf-8") as file:
                dati = json.load(file)

            frames_path = os.path.join( visual_file_path, "extractedFaceFrames") + "/"

            # -------------------- ESTRAZIONE EMOZIONI AUDIO --------------------
            logger.info("---------- ANALISI AUDIO " + video_name + " -------------")
            audio_file_path = os.path.join(audio_path, utils.config["General"]["temp_audio_name"])

            #la mia bellissima classe
            #ora imp-lementa VAD per ignorare il non parlato
            
            audioExtractor = AudioEmotionExtractor(
                vad_enabled=True,
                vad_rms_threshold=0.015,          # da tarare
                vad_fallback_to_neutral=False,     # oppure False per {}, FALSE NON FUNZIONA
                confidence_threshold=0.25         # pare andare bene cosí
            )
            audio_emotions = audioExtractor.predict_per_time_slot(
                audio_path=audio_file_path,
                total_ts=len(dati["time_slot"]),
                slot_seconds=2.0,
                context_seconds=6.0,   # opzionale, meglio 6.0
                centered=True,         # opzionale
                # --- dual-stream: ritorna anche embedding fuso per time-slot ---
                return_embedding=False #non ritorno piú i time slots
            )
            
            '''
            audioExtractor = AudioEmotionExtractor(
                vad_enabled=False,
                vad_rms_threshold=0.015,          # da tarare
                vad_fallback_to_neutral=False,     # oppure False per {}, FALSE NON FUNZIONA
                confidence_threshold=0.1         # pare andare bene cosí
            )

            #dovrebbe andare bene
            
            audio_emotions = audioExtractor.predict_per_time_slot(
                audio_path=audio_file_path,
                total_ts=len(dati["time_slot"]),
                slot_seconds=1.0,
                context_seconds=2.0,   # opzionale
                centered=True,         # opzionale
                # --- dual-stream: ritorna anche embedding fuso per time-slot ---
                return_embedding=False #non ritorno piú i time slots
            )
            '''

            '''
            audio_emotions = audioExtractor.predict_per_time_slot_spread(
                audio_path=audio_file_path,
                total_ts=len(dati["time_slot"]),
                slot_seconds=1.0,
                context_seconds=6.0,  # audio più lungo
                spread_k=3            # spalma su 3 slot (ts-1, ts, ts+1)
            )'''

            print("------------------------ ANALISI FACCIALE --------------------------")

            logger.info("------------------------ ANALISI FACCIALE --------------------------")

            # indicizza per lookup rapido (emotions + embedding)
            '''
            per non indicizzare gli embedding nel file json finale
            audio_emotions_by_ts = {
                x["time_slot"]: {"emotions": x.get("emotions", {}), "embedding": x.get("embedding", None)}
                for x in audio_emotions
            }'''
            audio_emotions_by_ts = {
                x["time_slot"]: {"emotions": x.get("emotions", {})}
                for x in audio_emotions
            }

            for slot in dati["time_slot"]:

                if slot["valid"]:
                    print(f"TS: {slot['ts']}")

                    # Richiamiamo il garbage collector ogni 5 time-slot per liberare memoria
                    if slot['ts'] % 5 == 0:
                        gc.collect()

                    frames_number = len(slot["frames"])
                    #print("Frame analizzati:", frames_number)

                    total = {
                        "angry": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0,
                        "happy": 0.0,
                        "neutral": 0.0,
                        "sad": 0.0,
                        "surprise": 0.0
                    }

                    emotions = list(total.keys())

                    for frame in slot["frames"]:
                        frame_val = cv2.imread(frames_path + frame)

                        all_detected_emotion =  emotionExtractor.extractFaceEmotion(image= frame_val) # pure qua usare nuova classe forse serve castina emozioni

                        if all_detected_emotion is None: # Caso limite in cui sul frame pre-processato face_mesh non riesce a trovare il volto
                            all_detected_emotion = torch.tensor([0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429])
    
                        #print(all_detected_emotion)
                        #print(all_detected_emotion.size())
                        all_detected_emotion_dict = { emotions[i]: all_detected_emotion[i].item() for i in range(len(emotions))  }

                        for k, v in all_detected_emotion_dict.items():
                            total[k] += v

                    # Media
                    average = {k: total[k] / frames_number for k in total}

                    # Normalizzazione
                    total_sum = sum(average.values())
                    normalized_data = {k: average[k] / total_sum for k in average}

                    # Cambio il nome della felicità da happy a joy per uniformità
                    normalized_data["joy"] = normalized_data.pop("happy")
                    normalized_data["anger"] = normalized_data.pop("angry")
                    normalized_data["sadness"] = normalized_data.pop("sad")

                    logger.info("Emozione dominante: " + max(normalized_data, key=normalized_data.get))
                    logger.debug(normalized_data)

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

                emotion_preds = pred_emo_from_omgdataset(df_segments=df_segments,model_name= text_model, dataset_name_train=dataset_train_text, utils=utils, audio_name=audio_file_path)

                logger.debug(emotion_preds.head())

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
                gt_ts = build_gt_with_intervals(
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
                    #audio_info = audio_emotions_by_ts.get(fe["time_slot"], {"emotions": {}, "embedding": None})
                    audio_info = audio_emotions_by_ts.get(fe["time_slot"], {"emotions": {}})
                    text_emotions = next((ts for ts in time_slots_text if ts["ts"] == fe["time_slot"]), None)
                    #emb = audio_info.get("embedding", None)
                    #if emb is not None:
                        #emb = emb.tolist()   # ndarray -> list (serializzabile JSON)
                    time_slots.append({
                        "ts": fe["time_slot"],
                        "ground_truth": text_emotions["ground_truth"] if text_emotions is not None else None,
                        "modal": {
                            "video": {
                                "emotions": to_python_float(fe["emotions"])
                            },
                            "audio": {
                                "emotions": to_python_float(audio_info.get("emotions", {})),
                            },
                            "text": {
                                "emotions": to_python_float(text_emotions["emotions"]) if text_emotions is not None else {}
                            }    # da integrare
                        }
                    })
                gc.collect()

            #caso altro dataset
            else:
                time_slots = []

                ground_truth_label = int(df_metadata[df_metadata["video"] == vid_name]["EmotionMaxVote"].values[0])
                gt_ts = build_gt_without_intervals(total_ts_video=len(dati["time_slot"]), video_label= ground_truth_label)

                for fe in face_emotions:

                    audio_info = audio_emotions_by_ts.get(fe["time_slot"], {"emotions": {}})
                    #audio_info = audio_emotions_by_ts.get(fe["time_slot"], {"emotions": {}, "embedding": None})
                    #emb = audio_info.get("embedding", None)
                    #if emb is not None:
                        #emb = emb.tolist()
                    time_slots.append({
                        "ts": fe["time_slot"],
                        "ground_truth": gt_ts[fe["time_slot"]],
                        "modal": {
                            "video": {
                                "emotions": to_python_float(fe["emotions"])
                            },
                            "audio": {
                                "emotions": to_python_float(audio_info.get("emotions", {})),
                            },
                            "text": {}    # da integrare
                        }
                    })
                gc.collect()


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
            #break

            # per fermare l'esecuzione quando facciamo test
            #if vid_name == "1v6f9b2KMRA.mp4":
            #   break

    feedback_results = generate_feedback_for_all_videos(complete_info_path=complete_info_path)

    with open(complete_info_path, "r", encoding="utf-8") as f:
        complete_info = json.load(f)

    for video_name, feedback_text in feedback_results.items():
        if video_name in complete_info:
            complete_info[video_name]["feedback"] = feedback_text

    with open(complete_info_path, "w", encoding="utf-8") as f:
        json.dump(complete_info, f, indent=4, ensure_ascii=False)

    # Calcolo le metriche per ogni stream
    stream_names = ["audio", "video", "text"]

    for stream in stream_names:
        evaluation_manager.evaluate_stream(stream_name=stream, video_name_list=name_list)

    evaluation_manager.evaluate_fusion(stream_names=stream_names, video_name_list=name_list)