from typing import TypedDict, TypeAlias
import numpy as np

#Класс объекта Detection содержит характеристики детекции
class Detection(TypedDict):
    frame: int
    visible: int
    x: float
    y: float

#Класс объекта Track содержит характеристики трека
class Track(TypedDict):
    frame_idx: list[int]
    x: list[float]
    y: list[float]
    pred_x: list[float]
    pred_y: list[float]
    size_frame: float
    size_sec: float
    dist_x: float
    dist_y: float
    first_frame: int
    last_frame: int
    confirmed: bool

#Класс объекта ALLTracks содержит все треки и их характеристики
ALLTracks: TypeAlias = dict[str, Track]

#Класс объекта CatBoostTrack содержит все треки и их новые характеристики после feature_generation
class CatBoostTrack(TypedDict):
    dist_x: float
    dist_y: float
    ratio: np.float64
    lenght: int
    std_x: np.float64
    std_y: np.float64
    speed_mae: np.float64
    mean_angle: np.float64
    line_rmse: np.float64
    line_max_error: np.float64
    parabola_rmse: np.float64
    mean_gap: np.float64
    ratio_gap: np.float64