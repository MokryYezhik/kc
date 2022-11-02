"""
Загрузка переменных окуружения из .env
"""

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

####################
# Подключение к БД #
####################

# Получение данных для подлкючения к БД
DB_INFO ={
    'database': os.environ.get('DB_DATABASE'),
    'host': os.environ.get('DB_HOST'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD'),
    'port': os.environ.get('DB_PORT')
}

# Названия таблиц БД
ITEMS_DB = 'public.post_text_df'
USERS_DB = 'public.user_data'
FEED_DB = 'public.feed_data'
ITEMS_FEATS_DB = 'public.posts_features_0'
MTE_CITY_DB = 'public.yaa_mte_city'

#############
# CB модель #
#############

# список признаков для CB модели в правильном порядке
__cb_model_feats_names = ['topic', 'total_tfidf', 'max_tfidf', 'mean_tfidf',
        'dist_1_cluster', 'dist_2_cluster', 'dist_3_cluster',
        'dist_4_cluster', 'dist_5_cluster', 'dist_6_cluster', 'dist_7_cluster',
        'dist_8_cluster', 'dist_9_cluster', 'dist_10_cluster', 'gender', 'age', 'country',
        'city', 'exp_group', 'os', 'source', 'hour', 'month', 'day']

# данные CB модели
CB_MODEL_INFO = {
    'model_name': 'catboost_model_example',
    'model_feats_names': __cb_model_feats_names,
}

#############
# NN модель #
#############

# список признаков для модели в правильном порядке
__nn_model_feats_names = ['total_tfidf_enc', 'max_tfidf',
       'mean_tfidf', 'dist_1_cluster', 'dist_2_cluster', 'dist_3_cluster',
       'dist_4_cluster', 'dist_5_cluster', 'dist_6_cluster', 'dist_7_cluster',
       'dist_8_cluster', 'dist_9_cluster', 'dist_10_cluster', 'gender', 'age_enc',
       'topic_covid', 'topic_entertainment', 'topic_movie', 'topic_politics',
       'topic_sport', 'topic_tech', 'exp_group_1', 'exp_group_2',
       'exp_group_3', 'exp_group_4', 'os_iOS', 'source_organic', 'month_11',
       'month_12', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
       'country_enc', 'city_enc', 'hour_enc']

# данные NN модели
NN_MODEL_INFO = {
    'model_name': 'ncf_CC_02_11688.pth',
    'model_feats_names': __nn_model_feats_names,
    'latent_size': 100,
    'feats_size': 38,
}

########################
# данные для энкодинга #
########################
__mte_country_dict = {
    'Russia': 0.10774984662537317,
    'Ukraine': 0.1704849812957921,
    'Belarus': 0.1844546919273293,
    'Azerbaijan': 0.08590941768511862,
    'Kazakhstan': 0.16265516638498748,
    'Finland': 0.13443072702331962,
    'Turkey': 0.2036526533425224,
    'Latvia': 0.16597077244258873,
    'Cyprus': 0.10802139037433155,
    'Switzerland': 0.08417508417508418,
    'Estonia': 0.12224448897795591
}
__mte_hour_dict = {
    14: 0.10301614273576891,
    7: 0.11252939837645096,
    15: 0.10205006899270648,
    19: 0.12902374172622527,
    6: 0.1319546568627451,
    10: 0.10629228946060629,
    13: 0.10516573803003117,
    20: 0.14066699473823674,
    21: 0.14237058502530908,
    11: 0.10579090110393827,
    16: 0.10142239895996635,
    17: 0.10406214284330628,
    22: 0.1434022625219218,
    18: 0.1042717099960638,
    8: 0.11236259482892089,
    9: 0.10872183372183372,
    23: 0.10738048930289693,
    12: 0.10336711839957484,
    0: 0.11460258364218187
}
ENCODES_INFO = {
    'mte_country_dict': __mte_country_dict,
    'mte_hour_dict': __mte_hour_dict,
    'mmscaler_user_age': (14, 95),
    'mmscaler_text_totaltfidf': (0, 19.00728108),
}




