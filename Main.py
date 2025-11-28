import os
import moviepy as movie
import cv2
import mediapipe as mp
import time
import numpy as np

"""
1. Estrazione audio da video
2. Normalizzazione audio
3. Trascrizione con Whisper, con timestamps
4. Analisi del testo per chunk 
5. Analisi audio da inizio chunk precedente ad inizio chunk successivo
6. Normalizzazione video e suddivisione in chunk di 1 sec 
7. Individuazione volto 
8. Analisi volto 
9. Allineamento analisi delle varie pipeline 
"""

# Funzione per estrarre le coordinate del volto con MediPipe e aggiunge un margine per creare un box ottimale 
def extract_face_box(detection, frame, margin=100):
    """
    Estrae le coordinate del volto da un detection Mediapipe
    e restituisce le coordinate del bounding box corrette con margine.
    """
    ih, iw, _ = frame.shape
    bbox = detection.location_data.relative_bounding_box

    x1 = int(bbox.xmin * iw)
    y1 = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    # Applica margine per includere bene il volto
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(iw, x1 + w +  margin )
    y2 = min(ih, y1 + h +  margin )

    return x1, y1, x2, y2

# Funzione per ruotare l'immagine attorno ad un centro specifico 
def rotate_image(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]

    # Se non specificato, centro immagine
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    return rotated

# Funzione per allineare il volto orizzontalmente basandosi sugli occhi
def align_face(frame, face_mesh):
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
    aligned = rotate_image(frame, -angle, center=center)
    return aligned


# Funzione per estrarre i singoli frame in scala di grigio con volto dal video e crea anche un video sempre in scala di grigio
# Viene fatto un down sampling = si prende solo un frame ogni 'frame_step' frames
def extract_face_frames(video, target_size=(224, 224), output_video_path: str = "faces_video.mp4", frame_step: int = 3, output_folder: str = "face_frames_extracted"):

    os.makedirs(output_folder, exist_ok=True)

    # Inizializzazione MediaPipe per face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Imposta video writer (grayscale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, 30, target_size, isColor=False)

    # Inizializzazione MediaPipe FaceMesh per allineamento
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)

    frame_count = 0
    saved_count = 0

    fps = 0
    if not frame_step:
        # Estrai frame rate del video e calcola frame_step custom
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_step = int(fps // 5)  

    # CLAHE per normalizzazione
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        # Downsampling -> da capire se crea problemi nel momento che si cerca di fare la fusione delle analisi testo, audio e video
        #                 dobbiamo tenere traccia ogni quanti frame stiamo pescando in modo da capire i chunk temporali del video
        # Salta i frame in base a frame_step    
        if frame_count % frame_step != 0:
            continue

        # Allineamento del volto
        aligned_frame = align_face(frame, face_mesh)

        # Face detection sul frame allineato
        rgb_frame = cv2.cvtColor(aligned_frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for det in results.detections:
                # Estrazione coordinate bounding box sul frame allineato
                x1, y1, x2, y2 = extract_face_box(det, aligned_frame)

                # BISOGNA AGGIUNGERE UN CONTROLLO, SE LE COORDINATE SONO TROPPO DISTANTI DA QUELLE ESTRATTE DAL 
                # PRECEDENTE ALLORA VA SCARTATO IL FRAM -> SIGNIFICA CHE POTREBBE AVER ESTRATTO MALE IL VOLTO 
                # OPPURE dovremmo fare la media di tutte le coordinate ed escludere quelli troppi distanti dalla media -> 
                #       in questo modo garantiamo una maggiore stabilità però se il soggetto # si muove tanto potrebbe essere complesso :(


                # Ritaglio del volto dal frame allineato
                face_crop = aligned_frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue  # sicurezza in caso di bbox errati

                # Scala di grigi
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                # Normalizzazione CLAHE
                gray = clahe.apply(gray)

                # Resize
                resized = cv2.resize(gray, target_size)

                # Salvataggio immagine
                saved_count += 1
                cv2.imwrite(f"{output_folder}/face_{saved_count:04d}.jpg", resized)

                # Salvataggio video risultatnte dal down sampling, grayscale, normalizzazione e ritaglio volto
                out_video.write(resized)

    face_detection.close()
    out_video.release()
    return frame_step, fps



# Funzione per l'estrazione della traccia audio da un video
def audioExtraction( videoPath: str ):
    # Create directory for audio analysis if it doesn't exist
    os.makedirs("AudioAnalisis", exist_ok=True)
    # Load the video file
    video = movie.VideoFileClip(videoPath)
    # Extract audio from the video
    audio = video.audio
    # Save the audio file
    audio_file_path = os.path.join("AudioAnalisis", "originalAudio.wav")
    audio.write_audiofile(audio_file_path)
    audio.close()
    video.close()
    return audio, video


VIDEO_FILE = r"CampioniVideo/redditi.mp4"

if __name__ == "__main__":

    print("=== ESTRAZIONE AUDIO DA VIDEO ===")
    audioExtraction( VIDEO_FILE )
    # Apro il video 
    video = cv2.VideoCapture(VIDEO_FILE)
    frame_step, fps = extract_face_frames( video )
    video.release()
    print("Nuovo video salvato")


