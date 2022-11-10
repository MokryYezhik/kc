from loguru import logger
from typing import List

from fastapi import FastAPI
import datetime


# Добавление модуля в PATH
# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).parent.parent))

from src.models import define_model
from src.crud import liked_posts, post_features, assign_feats, get_config
from src.schema import PostGet

app = FastAPI()


logger.info('loading config')
cfg = get_config()
logger.info('loading models')
model, predict_proba = define_model(cfg)
logger.info('service is up and running'.upper())


@app.get('/post/recommendations/', response_model=List[PostGet])
def recommend(id: int, time: datetime.datetime, limit: int) -> List[PostGet]:
    
    # загрузка фичей по пользователям
    user_posts_features = assign_feats(id, time)

    # предсказание
    logger.info('get predictions')
    pred_pobas = predict_proba(model, id, user_posts_features)
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


