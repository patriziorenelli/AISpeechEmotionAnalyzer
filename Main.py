import os
import moviepy as movie
import cv2
import mediapipe as mp
import time


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
def extract_face_box(detection, frame, margin=80):
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
    x2 = min(iw, x1 + w + margin)
    y2 = min(ih, y1 + h + margin)

    return x1, y1, x2, y2

# Funzione per estrarre i singoli frame in scala di grigio con volto dal video e crea anche un video sempre in scala di grigio
# Viene fatto un down sampling = si prende solo un frame ogni 'frame_step' frames
def extract_face_frames( video, target_size =(224, 224), output_video_path: str = "faces_video.mp4", frame_step: int = 3, output_folder:str = "face_frames_extracted" ):

    os.makedirs(output_folder, exist_ok=True)
    #Inizializzazione mediaPipe per face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    # Imposta video writer (grayscale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, 30, target_size, isColor=False)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1

        # Downsampling temporale, per selezionare solo 1 frame ogni 'frame_step' frames
        if frame_count % frame_step != 0:
            continue
        print(frame_count)
        # Converti in RGB per Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        results = face_detection.process(rgb_frame)

        if results.detections:
            for det in results.detections:
                # Ottengo le coordinate della box del volto
                x1, y1, x2, y2 = extract_face_box(det, frame)
                # Ritaglio il frame intorno al volto
                face_crop = frame[y1:y2, x1:x2]
                # Scala di grigi
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                # Resize del frame
                resized = cv2.resize(gray, target_size)

                # Salva immagine
                saved_count += 1
                cv2.imwrite(f"{output_folder}/face_{saved_count:04d}.jpg", resized)

                # Aggiungi al video
                out_video.write(resized)


    face_detection.close()
    out_video.release()



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
    extract_face_frames(video )
    video.release()
    print("Nuovo video salvato")


