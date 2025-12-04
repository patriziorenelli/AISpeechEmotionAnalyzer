import cv2
from deepface import DeepFace
from collections import Counter


# NOTE: Come accuratezza non è male, confonde emozioni quali paura e disgusto ad esempio e a volte rileva troppo neutral

# ATTENZIONE E' UN TEST SEMPLIFICATO PER CAPIRE LE PERFORMANCE DELLA CLASSIFICAZIONE AUTOMATICA TRRAMITE DeepFace
# Il campione video example_RAVDESS.mp4 è una clip sample dei video contenuti in RAVDESS con vari attori che dicono le varie frasi esprimento varie emozioni (1 per clip)

# Carica il video
# video_capture = cv2.VideoCapture("CampioniVideo/example_RAVDESS.mp4") # test con video sample RAVDESS originale
video_capture = cv2.VideoCapture("faces_video.mp4")  # test con video pre-processato poichè è molto più veloce grazie down sampling che faccio prima 

# Ottiene gli FPS del video
fps = video_capture.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

# Carica il classificatore Haar per il rilevamento del volto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0
emotion_list = []   # <-- SALVA le emozioni dominanti dei singoli frame

def format_time(seconds):
    """Converte i secondi in mm:ss"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1

    # Calcola timestamp del frame
    current_time_sec = frame_count / fps
    current_time_str = format_time(current_time_sec)

    # Converte il frame in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rileva i volti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Se viene rilevato un volto 
    if len(faces) > 0:

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        # Salva SOLO l’emozione dominante del frame
        emotion_list.append(result['dominant_emotion'])

    # Se il volto non è presente
    else:
        print(f"No face detected at {current_time_str} (frame {frame_count})")

    # Ogni 10 frame conta l'emozione dominante
    if frame_count % 10 == 0 and emotion_list:

        # Conteggio delle emozioni
        emotion_counts = Counter(emotion_list)

        # Emozione più frequente e quindi quella che dovrebbe identificare l'emozione espressa nel time slot 
        #   -> bisogna lavorare su questa parte per capire tecnica migliore per selezionare l'emozione più semplificativa e cercare di evitare errori
        top_emotion, top_count = emotion_counts.most_common(1)[0]

        print(f"\n⏱️ Time window: {format_time((frame_count-10)/fps)} → {current_time_str}")
        print(f"Emotions detected in frames {frame_count-9} to {frame_count}:")

        # Conteggio totale delle emozioni più espresse in ogni frame del time slot
        print(dict(emotion_counts))

        print(f"➡️ Final emotion for this window: **{top_emotion}** (count: {top_count})")
        print("-" * 60)

        emotion_list = []  # reset lista per la prossima finestra

video_capture.release()
