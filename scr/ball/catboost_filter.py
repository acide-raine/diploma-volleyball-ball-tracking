import catboost
from catboost import CatBoostClassifier
import numpy as np

from scr.shared.MyTypes import CatBoostTrack

def cat_boost_load(model_path: str) -> catboost.core.CatBoostClassifier:

    '''
    Функция инициализации модели

    :param model_path: str – путь к модели
    :return: model: catboost.core.CatBoostClassifier – ядро модели
    '''

    model = CatBoostClassifier()
    model.load_model(model_path)

    return model

def cat_boost_inference (model: catboost.core.CatBoostClassifier, catboost_track: CatBoostTrack) -> tuple[np.ndarray, np.ndarray]:

    '''
    Функция инференса модели

    :param model: catboost.core.CatBoostClassifier – ядро модели
           catboost_track: CatBoostTrack – словарь с характеристиками трека
    :return: prediction: np.ndarray – предсказанный класс (0 – не мяч, 1 – мяч)
             probability: np.ndarray – массив вероятностей по классам
    '''

    input = [list(catboost_track.values())]

    prediction = model.predict(input)
    probability = model.predict_proba(input)

    return prediction, probability

def cat_boost_filter(catboost_track: CatBoostTrack, model: catboost.core.CatBoostClassifier, P: float) -> bool:

    '''
    Функция фильтр треков

    :param catboost_track: CatBoostTrack – словарь с характеристиками трека
           model: catboost.core.CatBoostClassifier – ядро модели
           P: float – порог вероятности

    :return: bool: – подходит ли предсказание модели нужного класса под порог. True — мяч, False — не мяч
    '''

    prediction, probability = cat_boost_inference(model, catboost_track)

    return probability[0][1] >= P