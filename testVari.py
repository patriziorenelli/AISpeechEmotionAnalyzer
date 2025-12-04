# HSEmotion famiglia di modelli deep learning sviluppati 2020-2022 per emozioni dei volti, dovrebbe essere circa 30 % più accurato di DeepFace DA PROVARE 

# InsightFace con modello “emotion” si hanno vari modelli disponibili, in alcuni casi più performante di HSEmotion 
#               -> L'HO TESTATO TRAMITE IL CODICE SOTTO PRIMA CHE MI USCISSE UN CONFLIITTO DI LIBRERIE E VARI, SUL VIDEO DI TEST MI PARE SIA PEGGIORE DI DeepFace

# IL CODICE SOTTO ORA NON FUNZIONA A CAUSA DI CONFLITTI CON LIBRERIE E SIMILI, SE RIUSCITE AD AGGIUSTARLO PROVATELO :(

import cv2
import torch
from collections import Counter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# -------------------------------
# HSEmotion import
# -------------------------------
try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
except ImportError:
    HSEMOTION_AVAILABLE = False
    print("HSEmotion non installato, verrà ignorato.")

# -------------------------------
# Configurazione
# -------------------------------
MODEL_CHOICE = "hsemotion"  # "rafdb" o "hsemotion"
VIDEO_PATH = "faces_video.mp4"  # Percorso del video
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_STEP = 1  # Usa 1 per analizzare ogni frame, 2 per saltarne alcuni
AGGREGATE_EVERY = 10  # Numero di frame per aggregazione

_original_torch_load = torch.load

def torch_load_override(f, **kwargs):
    return _original_torch_load(f, weights_only=False)

# -------------------------------
# Selezione modello
# -------------------------------
if MODEL_CHOICE == "rafdb":
    MODEL_NAME = "adhityamw11/facial_emotions_image_detection_rafdb_microsoft_vit"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
elif MODEL_CHOICE == "hsemotion": # NON FUNZIONA AL MOMENTO
    if not HSEMOTION_AVAILABLE:
        raise ImportError("HSEmotion non disponibile. Installa `hsemotion` per usarlo.")
    # Puoi cambiare modello: 'enet_b0_8_best_afew', 'resnet18_afew', ecc.
    MODEL_NAME = "enet_b0_8_best_afew"

    # Caricamento forzato PER CONFLITTO CON PYTORCH 
    torch.load = torch_load_override
    fer = HSEmotionRecognizer(model_name=MODEL_NAME, device=DEVICE)
    torch.load = _original_torch_load

else:
    raise ValueError("Modello non valido. Scegli 'rafdb' o 'hsemotion'.")

# -------------------------------
# Funzioni
# -------------------------------
def predict_emotion(frame):
    """
    Prende un frame BGR da OpenCV e restituisce l'emozione più probabile.
    Funziona sia per rafdb sia per HSEmotion.
    """
    if MODEL_CHOICE == "rafdb":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            emotion = model.config.id2label[pred_idx]
        return emotion
    elif MODEL_CHOICE == "hsemotion":
        # HSEmotion si aspetta immagini RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(img, logits=True)
        return emotion

# -------------------------------
# Analisi video
# -------------------------------
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return

    frame_count = 0
    slot_count = 1
    emotions_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_STEP == 0:
            emotion = predict_emotion(frame)
            emotions_buffer.append(emotion)

        # Aggregazione ogni AGGREGATE_EVERY frame
        if len(emotions_buffer) == AGGREGATE_EVERY:
            counter = Counter(emotions_buffer)
            most_common = counter.most_common(1)[0][0]
            print(f"Time slot {slot_count}:")
            print(f"Frames analizzati: {AGGREGATE_EVERY}")
            print("Conteggio emozioni:", dict(counter))
            print("Emozione più frequente:", most_common)
            print("-" * 30)
            emotions_buffer = []
            slot_count += 1

        frame_count += 1

    # Eventuali frame rimanenti
    if emotions_buffer:
        counter = Counter(emotions_buffer)
        most_common = counter.most_common(1)[0][0]
        print(f"Time slot {slot_count}:")
        print(f"Frames analizzati: {len(emotions_buffer)}")
        print("Conteggio emozioni:", dict(counter))
        print("Emozione più frequente:", most_common)

    cap.release()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    analyze_video(VIDEO_PATH)
