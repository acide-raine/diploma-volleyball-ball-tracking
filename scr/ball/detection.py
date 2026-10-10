import onnxruntime as ort
import numpy as np
import cv2

from scr.shared.MyTypes import Detection


def load_model(model_path: str) -> ort.InferenceSession:
    '''

    Функция инициализация модели.

    :param model_path: str – путь к модели
    :return: session: ort.InferenceSession – "движок" модели
    '''

    session = ort.InferenceSession(model_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])

    return session

def model_params (session:ort.InferenceSession) -> tuple[str, str, int, list[int], list[int]]:
    '''

    :param session: ort.InferenceSession – "движок" модели
    :return: input_name: str – имя входного слоя
             output_name: str – имя выходного слоя
             output_seq: int – размерность выходного тензора
             input_shape: list[int] – размер входного тензора
             output_shape: list[int] – размер выходного тензора
    '''

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    output_seq = session.get_outputs()[0].shape[1]

    return input_name, output_name, output_seq, input_shape, output_shape

def preprocess_for_onnx(frame: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    '''

    Функция предобработки фрейма для нейросети:
    - перевод в серые оттенки
    – ресайз под INPUT_W, INPUT_H
    – перевод в float32 и нормализация в 0.0...1.0

    :param frame: np.ndarray – кадр
           input_w: int – ширина входящего кадра
           input_h: int – выоста входящего кадра

    :return: предобработанный кадр
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (input_w, input_h))

    return resize.astype(np.float32) / 255.0


def input_tensor(buffer: list[np.ndarray]) -> np.ndarray:
    '''

    Формирование входного тензора для нейросети

    :param buffer: список из 9 кадров
    :return: массив вида (1, 9, 288, 512) – 9 кадров, размером (512: 288) и с 1м батчем
    '''

    return np.expand_dims(np.stack((buffer), 0), 0)


def inference(session: ort.InferenceSession, input_name: str, output_name: str, preprocessed_frame: np.ndarray) -> np.ndarray:

    '''Функция инференса

    :param session: ort.InferenceSession – "движок" модели
           input_name: str – имя входного слоя
           output_name: str – имя выходного слоя
           preprocessed_frame: np.ndarray – предобработанный входной кадр
    :return: массив с предсказаниями модели (координаты, вероятности)
    '''

    return session.run([output_name], {input_name: preprocessed_frame})[0]


def heatmap_decoder (output: np.ndarray, trashold: float) -> tuple[int, int, int]:

    '''

    Декодер тепловой карты.
    Берет последний кадр из выходной последовательности (1, 9, n, n,)
    Ищет самые яркие пятна, отсекает все ниже trashold.
    Находит и обводит контуры. Если контура нет возвращает 0, -1, -1,
    Если есть, находит центр масс через moments, возвращает 1, x, y


    :param output: np.ndarray – выход из модели
           trashold: float – порог теплоты пятна

    :return: возвращает кортеж вида tpl = (int, int, int)
             tpl[0] - видимость мяча (1 == виден, 0 == нет)
             tpl[1] – х координата
             tpl[2] – у координата
    '''

    heatmap = output[0, -1, :, :]

    _, binary = cv2.threshold(heatmap, trashold, 1.0, cv2.THRESH_BINARY)
    countors, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not countors:
        return 0, -1, -1

    max_spot = max(countors, key=cv2.contourArea)
    M = cv2.moments(max_spot)

    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return 1, cx, cy

    else:
        return 0, -1, -1


def original_dimension(x: int, y: int, input_w: int, input_h: int, orig_w: int, orig_h: int) -> tuple[int, int]:
    '''
    Функция перевода предсказанных координат в пространстве рзмерностью [INPUT_W, INPUT_H],
    в пространство оригинального видео размерностью [orig_w, orig_h]

    :param x: int – предсказанная координата х
           y: int – предсказанная координата y
           input_w: int – ширина входа в нейросеть
           input_h: int – высота входа в нейросеть
           orig_w: int – оригинальная ширина
           orig_h: int – оригинальная высота

    :return: orig_x: int – предсказанный х в оригинальном пространстве
             orig_y: int – предсказанный у в оригинальном пространстве
    '''
    orig_x = int(x * orig_w/input_w)
    orig_y = int(y * orig_h/input_h)

    return orig_x, orig_y

def create_dict(frame_id: int, visible: int, x: int, y: int) -> Detection:
    '''

    Функция создания словаря с характеристиками детекции Detection (структура в файле MyTypes.py)

    :param frame_id: int – номер детектируемого фрейма
    :param visible: int – видимость мяча
    :param x: int – х координата
    :param y: int - у координатаа

    :return: detection: Detection – словарь с харатектристиками детекции
    '''


    detection = {"frame": frame_id,
                 "visible": visible,
                 "x": x,
                 "y": y}

    return detection