import numpy as np
import pandas as pd

from neighbor import Neighbor
from knn import KNN
from title_to_Tag import Title_to_tag

from data_util import *
from arena_util import load_json, write_json

### 1. data & preprocessing
### 1.1 load data
song_meta_path = 'res/song_meta.json'
train_path     = 'res/train.json'
val_path       = 'res/test.json'

song_meta = load_json(song_meta_path)
train     = load_json(train_path)

song_meta = pd.DataFrame(song_meta)
train     = pd.DataFrame(train)

### 1.2 only_title chage to tags
val = Title_to_tag(train_path=train_path, val_path=val_path).change()


### 1.3 convert "tag" to "tag_id"
tag_to_id, id_to_tag = tag_id_meta(train, val)
train = convert_tag_to_id(train, tag_to_id)
val   = convert_tag_to_id(val  , tag_to_id)


### 2. modeling : Neighbor
### 2.1 hyperparameters: pow_alpha, pow_beta
pow_alpha = 0.65
pow_beta  = 0.0

### 2.2 run Neighbor.predict() : returns pandas.DataFrame
pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, \
                train=train, val=val, song_meta=song_meta).predict()

### 3. modeling : KNN
### 3.1 hyperparameters: k, rho, weights
### 3.2 parameters: sim_songs, sim_tags, sim_normalize

song_k = 500
tag_k  = 90
song_k_step = 50
tag_k_step  = 10
rho = 0.4
weight_val_songs  = 0.9
weight_pred_songs = 1 - weight_val_songs
weight_val_tags   = 0.7
weight_pred_tags  = 1 - weight_val_tags
sim_songs = 'idf'
sim_tags  = 'idf'
sim_normalize = True

### 3.3 run KNN.predict() : returns pandas.DataFrame
pred = KNN(song_k=song_k, tag_k=tag_k, rho=rho, \
                   song_k_step=song_k_step, tag_k_step=tag_k_step, \
                   weight_val_songs=weight_val_songs, weight_pred_songs=weight_pred_songs, \
                   weight_val_tags=weight_val_tags, weight_pred_tags=weight_pred_tags, \
                   sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
                   train=train, val=val, song_meta=song_meta, pred=pred).predict()

### 4. post-processing
### 4.1 convert "tag_id" to "tag"
pred = convert_id_to_tag(pred, id_to_tag)
pred = generate_answers(load_json(train_path), to_list(pred))

write_json(pred, 'results.json')