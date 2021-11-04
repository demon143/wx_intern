import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../model'))
sys.path.append(os.path.join(BASE_DIR, '../'))

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import argparse
import warnings

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from mmoe import MMOE
from tensorflow.keras.optimizers import Adam, Adagrad
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# from tensorflow.python import keras
# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)


seed = 2021
print(seed)

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

warnings.filterwarnings('ignore')

targets = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'follow', 'comment']
sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']

dense_features = ['videoplayseconds'] + [f'fu_w2v_{i}' for i in range(10)]

vocab = {'userid': 250249, 'feedid': 112872, 'device': 3, 'authorid': 18789, 'bgm_song_id': 25160, 'bgm_singer_id': 17501}


epochs = 5
batch_size = 1024 * 4
embedding_dim = 256

train = pd.read_csv('data/user_action.csv')

feedid_embed = pd.read_csv('./data/fu_w2v.csv')
train = pd.merge(train , feedid_embed, on='feedid', how='left')

feed_info = pd.read_csv('./data/feed_info.csv')
train = pd.merge(train , feed_info, on='feedid', how='left')

fixlen_feature_columns = [DenseFeat(feat, 1) for feat in dense_features] + \
                         [SparseFeat(feat, vocabulary_size=vocab[feat], embedding_dim=embedding_dim) for feat in sparse_features] 
 

dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(dnn_feature_columns)


train_model_input = {name: train[name] for name in feature_names}

      
train_labels = [train[y].values for y in targets]

train_model = MMOE(dnn_feature_columns, num_tasks=len(targets), expert_dim=16, dnn_hidden_units=(256, 256),
                   tasks=['binary'] * len(targets), task_dnn_units=(128, 128))

train_model.compile("adagrad", loss='binary_crossentropy',)

for epoch in range(epochs):
    history = train_model.fit(train_model_input, train_labels,
                              batch_size=batch_size, epochs=1, verbose=1
                              )
    if epoch == 4:
        save_model(train_model, f'data/model_{epoch}_run{seed}.h5')
