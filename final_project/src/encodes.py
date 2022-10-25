"""
Загрузка и определение функция и словарей для кодирования признаков
"""
import pandas as pd
import psycopg2
from numpy import uint8

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from src.load_env import *


class OHE(BaseEstimator, TransformerMixin):
    """
    Класс для sklearn OneHotEncoder, который возвращает
    pd.DataFrame с названиями колонок
    """
    def __init__(self):
        self.ohe = OneHotEncoder(categories='auto',
                                 drop='first',
                                 dtype=uint8,
                                 sparse=False)


    def fit(self, X, y=None):
        self.ohe.fit(X)
        return self


    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = self.ohe.transform(X_)
        n_feats = self.ohe.n_features_in_
        feats = self.ohe.feature_names_in_
        cats = self.ohe.categories_

        columns = []
        for i in range(n_feats):
            main_cat = feats[i]
            cat_names = [f'{main_cat}_{name}' for name in cats[i][1:]]
            columns.extend(cat_names)

        df = pd.DataFrame(data=X_, columns=columns)
        return df



def min_max_scaler(X, X_min, X_max):
    """
    Кастомный MinMax Scaler, который масштабирует в наперед заданном
    интервале, полученном на обучающей выборке,
    на которой модель обучалась
    """
    X_ = X.values
    min, max = 0, 1
    X_std = (X - X_min) / (X_max - X_min)
    X_scaled = X_std * (max-min) + min
    return X_scaled


def __get_mte_city_encoding():
    '''
    Загрузка таблицы mte_city (MeanTargetEncoding для user_features['city'])
    '''
    conn = psycopg2.connect(**DB_INFO)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT *
        FROM  {MTE_CITY_DB}
        """
    )
    col_names = [desc[0] for desc in cursor.description]
    result = cursor.fetchall()
    df = pd.DataFrame(data=result, columns=col_names).drop(['index'], axis=1)
    mte_city_dict = {k: v for k, v in df.values}

    return mte_city_dict


def get_mte_dicts():
    """
    Функция возвращает
    - MeanTargetEncoding для user_features['country'];
    - MeanTargetEncoding для user_features['city'];
    - MeanTargetEncoding для feead['hour']
    """
    mte_country_dict = ENCODES_INFO['mte_country_dict']
    mte_city_dict = __get_mte_city_encoding()
    mte_hour_dict = ENCODES_INFO['mte_hour_dict']
                     
    return mte_country_dict, mte_city_dict, mte_hour_dict

mte_country_dict, mte_city_dict, mte_hour_dict = get_mte_dicts()