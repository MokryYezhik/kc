"""
Загрузка моделей и определение функций для предсказаний
"""

import os
import torch
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from torch import Tensor

from src.load_env import CB_MODEL_INFO, NN_MODEL_INFO
from src.crud import user_features, post_features

from pathlib import Path
PATH = Path(__file__).parent.parent.resolve()


"""
Определение словарей user_id --> index; post_id --> index
"""
__user_ids = np.unique(user_features['user_id'])
__post_ids = np.unique(post_features['post_id'])
user2id = {v: k for k, v in enumerate(__user_ids)}
post2id = {v: k for k, v in enumerate(__post_ids)}


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_cb_model():
    model_file = os.path.join(PATH, CB_MODEL_INFO['model_name'])
    model = CatBoostClassifier()
    model.load_model(model_file)
    return model

def cb_predict(model, feats):
    model_columns = CB_MODEL_INFO['model_feats_names']
    preds = model.predict_proba(feats[model_columns])[:, 1]
    return preds



class NCCF(torch.nn.Module):
    """
    Класс NN
    """
    def __init__(self, n_users, n_items, latent_size, feats_size):
        super().__init__()

        self.latent_size = latent_size
        self.feats_size = feats_size
        self.n_users = n_users
        self.n_items = n_items

        self.embedding_user = torch.nn.Embedding(self.n_users, latent_size)
        self.embedding_item = torch.nn.Embedding(self.n_items, latent_size)

        self.lin1 = torch.nn.Linear(latent_size*2 + feats_size, 64)  # 238
        self.lin2 = torch.nn.Linear(64, 1)

        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()

    def forward(self, users, items, feats):
        emb_users = self.embedding_user(users)
        emb_items = self.embedding_item(items)
        x = torch.concat([emb_users, emb_items, feats], dim=1)
        x = self.dropout(self.relu(self.lin1(x)))
        x = self.lin2(x)
        return x



def load_nn_model():
    """
    Загрузка и определение NN-модели
    """
    # model_file = './ncf_CC_01_17138.pth'
    model_file = os.path.join(PATH, NN_MODEL_INFO['model_name'])
    model_path = get_model_path(model_file)

    state_dict = torch.load(model_path)
    n_users = state_dict['embedding_user.weight'].shape[0]
    n_items = state_dict['embedding_item.weight'].shape[0]


    if n_users != len(user2id):
        raise Exception(f"Количество пользователей в БД ({len(user2id)}) и модели ({n_users}) НЕ совпадает")
    if n_items != len(post2id):
        raise Exception(f"Количество элементов в БД ({len(post2id)}) и модели ({n_items}) НЕ совпадает")
    
    model = NCCF(n_users,
                 n_items,
                 NN_MODEL_INFO['latent_size'],
                 NN_MODEL_INFO['feats_size'])

    model.load_state_dict(state_dict)
    return model


def nn_predict(model, id, features):
    """
    Предсказание вероятности для всех постов
    """
    model_columns = NN_MODEL_INFO['model_feats_names']
    feats = Tensor(features[model_columns].values)

    model.eval()
    posts = [post2id[post] for post in post_features['post_id'].values]
    posts = torch.LongTensor(posts)
    users = torch.LongTensor([user2id[id]] * len(post2id))

    with torch.no_grad():
        output = model(users, posts, feats)
        output = torch.sigmoid(output).flatten()

    return np.array(output)



def load_most_liked_posts():
    """
    Базовая модель - рекомендуем самые залайканные посты
    """
    model_file = './most_liked_posts.csv'
    model_path = get_model_path(model_file)
    most_liked_posts = pd.read_csv(model_path, sep=';')
    # posts_id = most_liked_posts['post_id'].values
    return most_liked_posts




