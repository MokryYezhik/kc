"""
Загрузка данных
"""
import psycopg2
import datetime
import pandas as pd
from loguru import logger
import os
import yaml

from src.load_env import *

from sqlalchemy import create_engine
from src.encodes import get_mte_dicts, OHE, min_max_scaler

from pathlib import Path
PATH = Path(__file__).parent.parent.resolve()

mte_country_dict, mte_city_dict, mte_hour_dict = get_mte_dicts()


def get_config():
    file_path = os.path.join(PATH, "params.yaml")
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def __batch_load_sql(query: str):
    """
    Загрузка таблицы по бачам
    """

    engine = create_engine(
        "postgresql://{}:{}@{}:{}/{}".\
            format(DB_INFO['user'],
                   DB_INFO['password'],
                   DB_INFO['host'],
                   DB_INFO['port'],
                   DB_INFO['database'])
    )

    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    generator_object = pd.read_sql(query, conn, chunksize=200000)
    for chunk_dataframe in generator_object:
        chunks.append(chunk_dataframe)
        logger.info(f'Got chunk: {len(chunk_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def __get_users_liked_posts():
    query = f"""
    SELECT DISTINCT post_id, user_id
    FROM {FEED_DB}
    WHERE action='like'
    """
    return __batch_load_sql(query)


def __get_post_features():
    '''
    загрузка таблицы post
    '''
    conn = psycopg2.connect(**DB_INFO)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT *
        FROM  {ITEMS_FEATS_DB}
        """
    )

    col_names = [desc[0] for desc in cursor.description]
    result = cursor.fetchall()
    df = pd.DataFrame(data=result, columns=col_names)

    df_ohe = OHE().fit(df[['topic']]).transform(df[['topic']])

    df = pd.concat([df, df_ohe], axis=1)#.drop(['topic'], axis=1)
    df['total_tfidf_enc'] = min_max_scaler(df['total_tfidf'], *ENCODES_INFO['mmscaler_text_totaltfidf'])
    return df


def __get_user_features():
    '''
    загрузка таблицы users
    '''
    conn = psycopg2.connect(**DB_INFO)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT *
        FROM  {USERS_DB}
        """
    )
    col_names = [desc[0] for desc in cursor.description]
    result = cursor.fetchall()
    df = pd.DataFrame(data=result, columns=col_names)

    ohe_cols = ['exp_group', 'os', 'source']
    df_ohe = OHE().fit(df[ohe_cols]).transform(df[ohe_cols])

    df = pd.concat([df, df_ohe], axis=1)#.drop(ohe_cols, axis=1)
    df['country_enc'] = df['country'].map(mte_country_dict)
    df['city_enc'] = df['city'].map(mte_city_dict)
    df['age_enc'] = min_max_scaler(df['age'], *ENCODES_INFO['mmscaler_user_age'])
    return df


def assign_feats(id: int, time: datetime.datetime):
    """
    Объединение признаков всех постов и данного пользователя,
    создание признаков по времени из запроса,
    кодирование признаков
    """

    # загрузка фичей по пользователям
    logger.info(f'user_id: {id}')
    logger.info('reading features')
    user_feats = user_features.loc[user_features['user_id'] == id]
    user_feats = user_feats.drop('user_id', axis=1)

    # загрузим фичи по постам
    logger.info('dropping columns')
    posts_feats = post_features.drop(['index', 'text'], axis=1)

    # объединим эти фичи
    logger.info('zipping everything')
    add_user_features = dict(zip(user_feats.columns, user_feats.values[0]))
    user_posts_features = posts_feats.assign(**add_user_features)

    # добавим информацию о дате рекомендаций
    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month
    user_posts_features['day'] = time.today().weekday() # day_of_week

    # кодирование фичей
    logger.info('preprocessing for nn model')

    user_posts_features['hour_enc'] = user_posts_features['hour'].map(mte_hour_dict)
    ohe_month = OHE().fit(pd.DataFrame(data=[10, 11, 12], columns=['month'])).\
                        transform(user_posts_features[['month']]).\
                        set_index(user_posts_features.index)
    ohe_day = OHE().fit(pd.DataFrame(data=[0, 1, 2, 3, 4, 5, 6], columns=['day'])).\
                        transform(user_posts_features[['day']]).\
                        set_index(user_posts_features.index)
    user_posts_features = pd.concat([user_posts_features, ohe_month, ohe_day], axis=1)

    user_posts_features = user_posts_features.set_index('post_id')

    return user_posts_features



logger.info('loading liked posts')
liked_posts = __get_users_liked_posts()
logger.info('loading users features')
user_features = __get_user_features()
logger.info('loading posts features')
post_features = __get_post_features()


