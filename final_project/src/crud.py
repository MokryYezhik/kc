"""
Загрузка данных
"""
import psycopg2
import pandas as pd
from loguru import logger

from src.load_env import *

from sqlalchemy import create_engine
from src.encodes import get_mte_dicts, OHE, min_max_scaler

mte_country_dict, mte_city_dict, mte_hour_dict = get_mte_dicts()


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
    df['total_tfidf'] = min_max_scaler(df['total_tfidf'], *ENCODES_INFO['mmscaler_text_totaltfidf'])
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

    df = pd.concat([df, df_ohe], axis=1).drop(ohe_cols, axis=1)
    df['country'] = df['country'].map(mte_country_dict)
    df['city'] = df['city'].map(mte_city_dict)
    df['age'] = min_max_scaler(df['age'], *ENCODES_INFO['mmscaler_user_age'])
    return df


logger.info('loading liked posts')
liked_posts = __get_users_liked_posts()
logger.info('loading users features')
user_features = __get_user_features()
logger.info('loading posts features')
post_features = __get_post_features()