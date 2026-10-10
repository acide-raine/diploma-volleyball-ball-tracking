import numpy as np

from scr.shared.MyTypes import Track, CatBoostTrack


def track_length (track: Track) -> int:

    '''
    Функция расчета длины трека

    :param track: Track – словарь с данными трека

    :return: length: int – длина трека

    '''

    length = len(track["x"])
    return length

def path_ratio(track: Track) -> np.float64:

    '''
    функция рассчета извилистости пути.
    Находит отношение полной длины пройденного пути к расстоянию
    по прямой между начальной и конечной точками траектории

    :param track: Track – словарь с данными трека
    :return: ratio: float – коэффициент извилистости
    '''

    x_arr = np.array(track["x"])
    y_arr = np.array(track["y"])

    d_x = np.diff(x_arr)
    d_y = np.diff(y_arr)

    path_length = np.sum(np.hypot(d_x, d_y))
    straight = np.hypot(x_arr[-1] - x_arr[0], y_arr[-1] - y_arr[0])

    ratio = path_length / (straight + 1e-6)

    return ratio

def mean_speed_calculate(track: Track) -> np.float64:

    '''
    Функция расчета средней скорости полета мяча в треке.

    :param track: Track – словарь с данными трека
    :return:speed_mean: np.float64 - средняя скорость
    '''

    d_x = np.diff(track["x"])
    d_y = np.diff(track["y"])
    d_t = np.diff(track["frame_idx"])

    speeds = np.hypot(d_x, d_y) / d_t

    speed_mean = np.mean(speeds)

    return speed_mean

def turn_angle(track: Track) -> np.ndarray:
    '''
    Вычисляет углы поворота траектории между последовательными шагами.

    Для каждой тройки последовательных точек функция строит два вектора
    движения и находит абсолютный угол между ними в градусах (от 0° до 180°).

    :param track: Track – словарь с данными трека
    :return: angles: bnp.ndarray – массив углов поворота в градусах.
    '''
    x_arr = np.array(track["x"])
    y_arr = np.array(track["y"])

    d_x = np.diff(x_arr)
    d_y = np.diff(y_arr)

    angles = []

    for i  in range(len(d_x) - 1):

        v1 = np.array([d_x[i], d_y[i]])
        v2 = np.array([d_x[i + 1], d_y[i + 1]])

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]

        angle = np.degrees(np.arctan2(abs(cross), dot))
        angles.append(angle)

    return np.array(angles)

def mean_angl(angles: np.ndarray) -> np.float64:
    '''
    Функция расчета среднего в массиве углов
    :param angles: np.ndarray – массив углов поворота в градусах.
    :return: mean_angle: np.float64 – средняя по массиву углов
    '''

    mean_angle = np.mean(angles)

    return mean_angle

def line_fit_rmse (track: Track) -> tuple[float, float]:

    '''
    Функция оценки отклонения траектории полета мяча от прямой линии

    Аппроксимирует точки трека прямой линией с помощью ортогональной
    регрессии. Затем рассчитывает ортогональные расстояния от каждой
    точки до этой линии и вычисляет метрики ошибок.

    :param track: Track – словарь с данными трека
    :return: rmse: float – среднеквадратичное отклонение
             max_error: float – максимальный единичный выброс
    '''

    x_arr = np.array(track["x"])
    y_arr = np.array(track["y"])

    poses = np.column_stack((x_arr, y_arr))

    centroid = poses - np.mean(poses, axis=0)

    cov = np.cov(centroid.T)
    values, vectors = np.linalg.eig(cov)
    normal = vectors[:, np.argmin(values)]

    distance = np.abs(np.dot(centroid, normal))
    rmse = np.sqrt(np.mean(distance**2))
    max_error = np.max(distance)

    return rmse, max_error

def parabola_fit_rmse(track: Track) -> np.float64:

    '''
    Функция оценивает отклонение полета мяча от балистической параболы

    Аппроксимирует изменение координат X и Y во времени (по кадрам)
    полиномами второй степени. Используется для валидации полета мяча:
    низкий RMSE подтверждает естественное свободное падение под действием гравитации.

    :param track: Track – словарь с данными трека
    :return: rmse: float – среднеквадратичное отклонение
    '''

    x_arr = np.array(track["x"])
    y_arr = np.array(track["y"])
    all_frame = np.array(track["frame_idx"])

    polly_x = np.polyfit(all_frame, x_arr, deg = 2)
    polly_y = np.polyfit(all_frame, y_arr, deg = 2)

    x_pred = np.polyval(polly_x, all_frame)
    y_pred = np.polyval(polly_y, all_frame)

    distance = np.sqrt((x_arr - x_pred) ** 2 + (y_arr - y_pred) ** 2)

    rmse = np.sqrt(np.mean(distance**2))

    return rmse

def mean_ratio_gap(track: Track) -> tuple[np.float64, np.float64]:

    '''
    Функция определяет, насколько непрерывным был трек во времени, вычисляя средний
    интервал между детекциями и долю переходов, где кадры были потеряны.

    :param track: Track – словарь с данными трека
    :return: mean_gap: среднее количество кадров между соседними детекциями.
             ratio_gap: доля переходов с потерями
    '''
    all_frame = np.array(track["frame_idx"])

    diff = np.diff(all_frame)

    mean_gap = np.mean(diff)
    ratio_gap = np.sum(diff > 1) / len(diff)

    return mean_gap, ratio_gap

def std_cords (track: Track) -> tuple[np.float64, np.float64]:
    '''
    Вычисляет стандартное отклонение координат трека по осям X и Y.
    Показывает общую амплитуду и масштаб перемещения объекта на видео.

    :param track: Track – словарь с данными трека
    :return: std_x: np.float64 – среднеквадратичное отклонение по оси X (ширина разброса).
             std_y: np.float64 – среднеквадратичное отклонение по оси Y (высота разброса).
    '''
    std_x = np.std(track["x"])
    std_y = np.std(track["y"])

    return std_x, std_y

def feature_generator(track: Track) -> CatBoostTrack:
    '''
    Функция генерации признаков и сбора словаря CatBoostTrack

    :param track: Track – словарь с данными трека
    :return: extended_track: CatBoostTrack – словарь с обновленными данными трека
    '''

    extended_track = track.copy()

    lenght = track_length(track)
    std_x, std_y = std_cords(track)

    ratio = path_ratio(track)

    speed_mae = mean_speed_calculate(track)
    mean_gap, ratio_gap = mean_ratio_gap(track)

    angle = turn_angle(track)
    mean_angle = mean_angl(angle) if len(angle) > 0 else 0.0

    line_rmse, line_max_error = line_fit_rmse(track)
    parabola_rmse = parabola_fit_rmse(track) if lenght >= 3 else 0.0

    extended_track["lenght"] = lenght
    extended_track["std_x"] = std_x
    extended_track["std_y"] = std_y

    extended_track["ratio"] = ratio

    extended_track["speed_mae"] = speed_mae

    extended_track["mean_angle"] = mean_angle

    extended_track["line_rmse"] = line_rmse
    extended_track["line_max_error"] = line_max_error

    extended_track["parabola_rmse"] = parabola_rmse

    extended_track["mean_gap"] = mean_gap
    extended_track["ratio_gap"] = ratio_gap

    trash_keys = ["x", "y", "pred_x", "pred_y", "size_sec", "size_frame",
                  "first_frame", "last_frame", "frame_idx", "confirmed", "miss_counter"]

    for key in list(extended_track.keys()):
        if key in trash_keys:
            extended_track.pop(key)

    return extended_track