# DA SCARICARE VIDEO E POI ANALIZZARE VOLTO, VOCE E TESTO
from  Utilities.utils import *
import cv2
from Preprocessing.Video.preprocessing_video import *
import shutil
import json
from PIL import Image
from Pipeline.Video.EmotionExtractor import *


# Questo codice è compatibile sia con Ravdess che OMGEmotion
if __name__ == "__main__":
    config = json.load(open("config.json"))
    utils = Utils(config)
    REPO_ID = "PiantoDiGruppo/Ravdess_AML"
    #REPO_ID = "PiantoDiGruppo/OMGEmotion_AML"
    name_list = utils.get_file_list_names(REPO_ID)

    BASE_PATH = "CompletePipeline"

    for video_name in name_list:
        print("------ANALISI VIDEO " + video_name +" ---------")
        if "json" not in  video_name:

            vid_name = video_name.split("/")[1]
            general_path = config["Paths"]["base_path"] +  "/" +  vid_name
            visual_file_path = general_path + "/Video"  

            utils.download_single_video_from_hug(REPO_ID, video_name, visual_file_path )

            # -------------------- Qui fare operazioni di processing ed analisi  ---------------------------------

            prep = PreprocessingVideo()
            video_path = visual_file_path + "/" + vid_name
            prep.extract_face_frames_HuggingVersion(video = cv2.VideoCapture(video_path), video_name = vid_name, output_folder =  visual_file_path + "/extractedFaceFrames" )

            audio_path = general_path + "/Audio"  

            utils.audioExtraction(video_path, output_path = audio_path)

            # per ogni time slot ora bisogna enalizzare le emozioni

            json_file_path = visual_file_path + "/extractedFaceFrames/info.json"

            emotionExtractor = EmotionExtractor()

            # Apre il json contenente le informazioni sui time slot e frame 
            with open(json_file_path, 'r', encoding='utf-8') as file:
                dati = json.load(file)
                frames_path = visual_file_path + "/extractedFaceFrames/"

                # Analizziamo i singoli time slot 
                for slot in dati['time_slot']:
                    # ci sta il campo valid dentro -> usare quello 

                    if len(slot['frames']) != 0:
                        print(f"TS: {slot['ts']}")

                        frames_number = len(slot['frames'])

                        print("Frame analizzati: " + str(frames_number) )
                        
                        total = {k: 0.0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}

                        for frame in slot['frames']:
                            frame_val =  cv2.imread(frames_path+frame)
                            dominant_emotion, all_detected_emotion = emotionExtractor.extractFaceEmotion( frame_val ) 

                            #print(dominant_emotion)
                            #print(all_detected_emotion)

                            for k, v in all_detected_emotion.items():
                                total[k] += v

                        # TODO Qui calcola per ogni emozione la somma dei suoi score diviso il numero di frame analizzati 
                        average = {k: total[k] / frames_number for k in total}
                            
                        # Normalizzo gli score
                        total_sum = sum(average.values())
                        normalized_data = {key: value / total_sum for key, value in average.items()}
                        
                        print("Emozione predetta maggiormente nel time slot: " + max(normalized_data, key=normalized_data.get))
                        print(normalized_data)

                        print(".........")  





            # --------------------------------------------------------------------------------------------------
            # FARE: cancellazione cartella e creazione di un file json con nome video, i singoli time slot e sezione video, testo, audio e per ogni time slot json emozioni


            # elimina la cartella con tutti i file creati
            #shutil.rmtree(BASE_PATH) 
            #break # messo per limitare il numero di file da analizzare durante la fase di sviluppo 