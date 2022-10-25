from loguru import logger
from typing import List
import pandas as pd

from fastapi import FastAPI
import datetime

from torch import Tensor
from pathlib import Path


# Добавление модуля в PATH
# import sys
# sys.path.append(str(Path(__file__).parent.parent))

from src.load_env import NN_MODEL_INFO
from src.encodes import OHE, mte_hour_dict
from src.models import load_model, predict
from src.crud import liked_posts, user_features, post_features
from src.schema import PostGet

app = FastAPI()


logger.info('loading model')
model = load_model()
logger.info('service is up and running'.upper())


@app.get('/post/recommendations/', response_model=List[PostGet])
def recommend(id: int, time: datetime.datetime, limit: int) -> List[PostGet]:
    # загрузка фичей по пользователям
    logger.info(f'user_id: {id}')
    logger.info('reading features')
    user_feats = user_features.loc[user_features.user_id == id]
    user_feats = user_feats.drop('user_id', axis=1)

    # загрузим фичи по постам
    logger.info('dropping columns')
    posts_feats = post_features.drop(['index', 'text'], axis=1)

    # объединим эти фичи
    logger.info('zipping everything')
    add_user_features = dict(zip(user_feats.columns, user_feats.values[0]))
    user_posts_features = posts_feats.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # добавим информацию о дате рекомендаций
    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month
    user_posts_features['day'] = time.today().weekday() # day_of_week

    # кодирование фичей
    logger.info('preprocessing for model')
    user_posts_features['hour'] = user_posts_features['hour'].map(mte_hour_dict)

    ohe_month = OHE().fit(pd.DataFrame(data=[10, 11, 12], columns=['month'])).\
                        transform(user_posts_features[['month']]).\
                        set_index(user_posts_features.index)
    ohe_day = OHE().fit(pd.DataFrame(data=[0, 1, 2, 3, 4, 5, 6], columns=['day'])).\
                        transform(user_posts_features[['day']]).\
                        set_index(user_posts_features.index)

    user_posts_features = pd.concat([user_posts_features, ohe_month, ohe_day], axis=1)
    model_columns = NN_MODEL_INFO['nn_model_feats_names']
    user_posts_features = Tensor(user_posts_features[model_columns].values)

    # предсказание
    logger.info('get predictions')
    pred_pobas = predict(model, id, user_posts_features)
    post_features['pred'] = pred_pobas

    # фильтрация уже понравивишихся постов
    user_liked_posts = liked_posts[liked_posts['user_id'] == id]['post_id'].values
    filtered = post_features[~post_features['post_id'].isin(user_liked_posts)]

    # предсказание по вероятности
    logger.info('get top n')
    recommended_posts = filtered.sort_values('pred', ascending=False)[:limit]

    return [
        PostGet(**{
            'id': recommended_posts.iloc[i]['post_id'],
            'text': recommended_posts.iloc[i]['text'],
            'topic': recommended_posts.iloc[i]['topic']
        }) for i in range(recommended_posts.shape[0])
    ]



