import cv2
from deepface import DeepFace


# QUI DENTRO FARE QUALCOSA CHE ANALIZZA I FRAME COMPRESI NEL JSON E IDENTIFICA LE EMOZIONI -> PER OGNI TIME SLOT RESTITUISCE DIZIONARIO CON LE EMOZIONI IDENTIFICATE E IL LORO SCORE 

class EmotionExtractor:

    
    # Analizza le emozioni di un singolo frame
    def extractFaceEmotion(self, frame, return_all=True): # se return_all = True ritorna tutte le probabilità delle emozioni

        
        try:
            result = DeepFace.analyze( frame, actions=['emotion'], enforce_detection=False )

            #print(result)
            
            # Se DeepFace produce una lista, prendiamo il primo elemento
            if isinstance(result, list):
                result = result[0]

            dominant_emotion = result["dominant_emotion"]
            emotion_scores = result["emotion"]

            if return_all:
                return dominant_emotion, emotion_scores
            
            return dominant_emotion

        except Exception as e:
            print(f"[ERROR] Emotion analysis failed: {e}")
            return None