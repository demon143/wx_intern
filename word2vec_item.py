from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import multiprocessing
import tqdm
import os



def w2v_feat(df, feat, length, w2v_model_path):
    if os.path.exists(w2v_model_path):
        print('find existing w2v model: {}'.format(w2v_model_path))
        model = Word2Vec.load(w2v_model_path)
    else:
        print('start word2vec')
        model = Word2Vec(df[feat].values, vector_size=length, window=5, min_count=1, sg=1, hs=0, negative=5,
                        epochs=10, seed=1)
        model.save(w2v_model_path)
    return model

def get_user_feed_w2v(train_path, w2v_path, save_path, num):
    user_action = pd.read_csv(train_path)
    user_action['feedid'] = user_action['feedid'].astype(str)

    feed_list = user_action['feedid'].unique()

    user_action = user_action.groupby('userid')['feedid'].agg(list).reset_index()

    model = w2v_feat(user_action, 'feedid', num, w2v_path)

    tmp_list = []
    for x in feed_list:
        tmp_list.append(model.wv[x])
    tmp_list = pd.DataFrame(tmp_list, columns=[f'fu_w2v_{w}' for w in range(num)])
    tmp_list['feedid'] = feed_list
    tmp_list['feedid'] = tmp_list['feedid'].astype(int)
    tmp_list.to_csv(save_path, index = None)

if __name__ == "__main__":
    user_action = './data/user_action.csv'
    fu_w2v_path = './data/user_feed.w2v'
    feed_w2v_path = './data/fu_w2v.csv'
    get_user_feed_w2v(user_action, fu_w2v_path, feed_w2v_path, 10)

