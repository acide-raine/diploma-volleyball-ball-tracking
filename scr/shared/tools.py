import cv2
import numpy as np
import time
import sys

from scr.shared.MyTypes import Detection, Track

def track_drawing(frame: np.ndarray, track: Track, track_id: int) -> np.ndarray:

    '''
    Функция отрисовки трека
    '''

    output = frame.copy()

    cv2.circle(output, (int(track["x"][-1]), int(track["y"][-1])), 15, (0, 255, 0), 3)

    cv2.putText(output, str(track_id),(10, 30)
                ,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)

    return output

def detect_drawing(frame: np.ndarray, detect: Detection) -> np.ndarray:

    '''
    Функция отрисовки детекции
    '''

    output = frame.copy()

    cv2.circle(output, (int(detect["x"]), int(detect["y"])), 10, (0, 0, 255), 2)

    return output

def court_drawing(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:

    '''
    Функция отрисовки углов и контура площадки
    '''

    output = frame.copy()
    corners = corners.astype(np.int32)

    cv2.polylines(output, [corners], True, (0, 255, 0), 3)

    for i, (x, y) in enumerate(corners):
        cv2.circle(output, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(output, str(i), (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return output

def court_drawing(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:

    '''
    Функция отрисовки углов и контура площадки
    '''

    output = frame.copy()
    corners = np.array(corners, dtype=np.int32)

    cv2.polylines(output, [corners], True, (0, 255, 0), 3)

    for i, (x, y) in enumerate(corners):
        cv2.circle(output, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(output, str(i), (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return output

def progress_bar_drawing(current_frame, total_frames, start_time):

    '''
    Функция отрисовки прогресс бара в консоль
    '''

    bar_length = 30
    progress = current_frame / total_frames
    block = int(round(bar_length * progress))

    bar_text = "█" * block + "░" * (bar_length - block)
    percent = int(progress * 100)

    current_time = time.time()
    elapsed_minutes = round((current_time - start_time) / 60, 2)
    video_minutes = round(total_frames / 30 / 60, 2)

    output = (f"\r\033[K[{bar_text}] {percent}% | На обработку {video_minutes} минут видео ушло {elapsed_minutes} минут")

    sys.stdout.write(output)
    sys.stdout.flush()

def convert_numpy(obj):

    '''
    Функция конвертации нампай объектов в питоновские типы для дампа в джисон
    '''

    if hasattr(obj, "dtype"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj