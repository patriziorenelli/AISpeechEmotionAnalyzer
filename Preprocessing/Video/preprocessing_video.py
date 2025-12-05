import cv2

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

print(preprocessing_video(r"D:/Magistrale/Sapienza/ComputerScience/Advanced Machine Learning/AISpeechAnalyzer/AISpeechEmotionAnalyzer/testvideo.mp4"))

