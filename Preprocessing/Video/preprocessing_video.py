import cv2
import os 
import mediapipe as mp 
import numpy as np
import json

class PreprocessingVideo:
    # Funzione per estrarre le coordinate del volto con MediPipe e aggiunge un margine per creare un box ottimale 
    def extract_face_box(self, detection, frame, margin=50):
            """
            Estrae le coordinate del volto da un detection Mediapipe
            e restituisce le coordinate del bounding box corrette con margine.
            """
            ih, iw, _ = frame.shape
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * iw) # in basso a sx
            y1 = int(bbox.ymin * ih) # in basso a sx
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            x3 = min(iw, x1)
            y3 = min(ih, y1)

            # Applica margine per includere bene il volto
            x1 = max(0, x1 - margin )
            y1 = max(0, y1 - margin)

            x4 = min(iw, x3 + w + margin )
            y4 = min(ih, y3 + h + margin)


            return x1, y1,x4,y4

    # Funzione per ruotare l'immagine attorno ad un centro specifico 
    def rotate_image(self, image, angle, center=None, scale=1.0):
            h, w = image.shape[:2]

            # Se non specificato, centro immagine
            if center is None:
                center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

            return rotated

    # Funzione per allineare il volto orizzontalmente basandosi sugli occhi
    def align_face(self, frame, face_mesh):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                return frame

            landmarks = res.multi_face_landmarks[0]
            ih, iw, _ = frame.shape

            left_eye = landmarks.landmark[33] # occhio sx
            right_eye = landmarks.landmark[263] # occhio dx

            x1, y1 = int(left_eye.x * iw), int(left_eye.y * ih)
            x2, y2 = int(right_eye.x * iw), int(right_eye.y * ih)

            # Calcolo angolo tra occhi
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Filtra angoli estremi
            if abs(angle) > 30:
                return frame

            # Ruota attorno al centro degli occhi
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            aligned = self.rotate_image(frame, -angle, center=center)
            return aligned


    def extract_face_frames_HuggingVersion(self, video, video_name="video", target_size=(224, 224), frame_step: int = 5, output_folder: str ="Prove/prep-train/face_frames_extracted"):

            os.makedirs(output_folder, exist_ok=True)

            # Cartella base del video
            #video_base_folder = os.path.join(output_folder, os.path.splitext(os.path.basename(video_name))[0])
            #os.makedirs(video_base_folder, exist_ok=True)

            # Creazione json per info 
            json_content = {
                #"file_name": video_base_folder,
                "file_name":os.path.splitext(os.path.basename(video_name))[0],
                #"emotion": video_base_folder.split("-")[2], questo era valido su Ravdess
                "time_slot": [ ]  } 

            # Dizionario temporaneo: ts → lista frame salvati
            time_slot_frames = {}

            # Inizializzazione MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

            # FaceMesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True, 
                max_num_faces=1, 
                refine_landmarks=True, 
                min_detection_confidence=0.3
            )

            frame_count = 0
            saved_count = 0

            fps = int(video.get(cv2.CAP_PROP_FPS))
            
            """
            fps = 30 → frame_step = max(1, 6) = 6 → estraggo 1 frame ogni 6 (5 fps).
            fps = 25 → frame_step = 5 → estraggo 25/5 = 5 fps.
            fps = 3 → 3 // 5 = 0 → max(1,0) = 1 → estraggo ogni frame (3 fps).
            """
            if not frame_step:
                frame_step = max(1, int(fps // 5))
            #print(frame_step)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                frame_count += 1

                # Skip in base al frame_step
                if frame_count % frame_step != 0:
                    continue

                aligned_frame = self.align_face(frame, face_mesh)
                rgb_frame = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                # Calcolo time-slot attuale
                time_slot = (frame_count - 1) // fps + 1

                if results.detections:

                    for det in results.detections:
                        x1, y1, x4, y4 = self.extract_face_box(det, aligned_frame)

                        face_crop = aligned_frame[y1:y4, x1:x4]
                        if face_crop.size == 0:
                            continue

                        #DeepFace ha usato immagini RGB per l'addestramento quindi meglio usarle senza scala di grigi
                        #gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        #gray = clahe.apply(gray)
                        #resized = cv2.resize(gray, target_size)


                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        # Converti in LAB, applica CLAHE solo al canale L, poi torna a BGR
                        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        l_clahe = clahe.apply(l)
                        lab_clahe = cv2.merge((l_clahe, a, b))
                        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

                        # ridimensiona all'input richiesto
                        resized = cv2.resize(enhanced, target_size)

                        # Nome file
                        file_name = f"face_ts{time_slot}_fr{frame_count}.jpg"
                        full_path = os.path.join(output_folder, file_name)

                        cv2.imwrite(full_path, resized)
                        saved_count += 1

                        # Inserimento nel dizionario temporaneo
                        if time_slot not in time_slot_frames:
                            time_slot_frames[time_slot] = []

                        time_slot_frames[time_slot].append(file_name)

                else:
                    print(f"No face detected at frame {frame_count}")

            face_detection.close()

            # --- CREAZIONE FINALE DEL JSON ---
            total_ts = ((frame_count - 1) // fps) + 1

            for ts in range(1, total_ts + 1):
                if ts in time_slot_frames:
                    frames = time_slot_frames[ts]
                    json_content["time_slot"].append({
                        "ts": ts,
                        "valid": True,
                        "frames": frames
                    })
                else:
                    # Time-slot senza frame → invalid
                    json_content["time_slot"].append({
                        "ts": ts,
                        "valid": False,
                        "frames": []
                    })

            # Salvataggio JSON
            json_path = os.path.join(output_folder, "info.json")
            with open(json_path, "w") as f:
                json.dump(json_content, f, indent=4)

            print(f"\nJSON creato in: {json_path}")
            print("Time-slot trovati:", len(time_slot_frames))

            return fps  



# ---------------------- CODICE GIA' PRESENTE NEL FILE ----------------------
def frame_generator(video_path, fps_target=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossibile aprire il video")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_target is not None:
        interval = video_fps / fps_target
    else:
        interval = 1  # prendi tutto

    next_frame = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= next_frame:
            yield frame
            next_frame += interval

        frame_idx += 1

    cap.release()

def preprocessing_video(video_path: str) -> int:
    """
    Preprocess the video for the pipeline. These preprocessing steps include:
    
    - Rimozione dei frame “non informativi”:
        - Frame sfocati - Sharpness score
        - Frame senza volto - Face detection: MediaPipe Face Detection, RetinaFace 
        - Frame con landmark non rilevabili
    - Scelta dei frame più espressivi (Calcolando variazioni tra landmark). 
        Paper: https://www.mdpi.com/2076-3417/9/21/4678
    - Uniform sampling

    Parameters
    ----------
    video_path (str): 
        The path to the video file.

    Returns
    ----------
    frame_list (list): 
        A list containing the preprocessed frame of video.
    """
    i=0

    for frame in frame_generator(video_path, fps_target=10):
       if i==0:
          cv2.imwrite("frame0.jpg", frame)
       i += 1
    
    return i
#print(preprocessing_video(r"D:/Magistrale/Sapienza/ComputerScience/Advanced Machine Learning/AISpeechAnalyzer/AISpeechEmotionAnalyzer/testvideo.mp4"))

'''
from datasets import load_dataset
from huggingface_hub import login

login(token="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")

ds = load_dataset(
    "PiantoDiGruppo/AMLDataset2",
    split="train",
    streaming=True
)

for row in ds:
    print(row) 
    break  # processa una riga alla volta
'''




#import torch
#print("Versione Torch:", torch.__version__)
#print("CUDA disponibile:", torch.cuda.is_available())
#print("CUDA version:", torch.version.cuda)