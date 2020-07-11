import numpy as np
import pandas as pd
import os
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm

class Neighbor:
    '''
    Neighbor-based Collaborative Filtering
    > Neighbor-tag_beta1.0 Version Update
      + tag prediction
    '''

    __version__ = "Neighbor-tag_beta1.0"

    def __init__(self, pow_alpha, pow_beta, train=None, val=None, \
                 verbose=True, version_check=True):
        '''
        pow_alpha, pow_beta : float (0<= pow_alpha, pow_beta <= 1)
        train, val : pandas.DataFrame
        verbose : boolean; True(default)
        version_check : boolean; True(default)
        '''

        self.train_id = train["id"].copy()
        self.train_tags = train["tags"].copy()
        del train

        self.val_id = val["id"].copy()
        self.val_tags = val["tags"].copy()
        del val
        
        self.pow_alpha = pow_alpha
        self.pow_beta = pow_beta

        self.verbose = verbose
        self.__version__ = Neighbor.__version__

        if version_check:
            print(f"Neighbor version: {Neighbor.__version__}")

        if not (0 <= self.pow_alpha <= 1):
            raise ValueError('pow_alpha is out of [0,1].')
        if not (0 <= self.pow_beta <= 1):
            raise ValueError('pow_beta is out of [0,1].')

        # TOTAL_TAGS = 30653  # total number of tags in train, val, tags
        TOTAL_TAGS = 30197  # in train, val

        freq_tags = np.zeros(TOTAL_TAGS, dtype=np.int64)
        for _tags in self.train_tags:
            freq_tags[_tags] += 1
        
        self.freq_tags_powered_beta = np.power(freq_tags, self.pow_beta)
        self.freq_tags_powered_another_beta = np.power(freq_tags, 1-self.pow_beta)
            

    def predict(self, start=0, end=None, auto_save=False, auto_save_step=500, auto_save_fname='auto_save'):
        '''
        start, end : range(start, end). if end = None, range(start, end of val)
        auto_save : boolean; False(default)
        auto_save_step : int; 500(default)
        auto_save_fname : string (without extension); 'auto_save'(default)
        @returns : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''

        # TODO: Remove unsupported module 'tqdm'.
        if end:
            _range = tqdm(range(start, end)) if self.verbose else range(start, end)
        elif end == None:
            _range = tqdm(range(start, self.val_id.index.stop)) if self.verbose else range(start, self.val_id.index.stop)

        pred = []
        all_tags = [set(tags) for tags in self.train_tags]  # list of set

        # TODO: use variables instead of constants
        # TOTAL_TAGS      = 30653   # total number of tags
        TOTAL_TAGS = 30197  # in train, val
        MAX_TAGS_FREQ   = 16465   # max freqency of tags for all playlists in train data
        TOTAL_PLAYLISTS = 115071  # total number of playlists

        for uth in _range:
            
            playlist_tags = set(self.val_tags[uth])
            playlist_size = len(playlist_tags)

            if len(playlist_tags) == 0:
                pred.append({
                    "id" : int(self.val_id[uth]),
                    "songs" : [],
                    "tags" : []
                })
                if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                    self._auto_save(pred, auto_save_fname)
                continue

            track_feature = {track_i : {} for track_i in range(TOTAL_TAGS)}
            relevance = np.concatenate((np.arange(TOTAL_TAGS).reshape(TOTAL_TAGS, 1), np.zeros((TOTAL_TAGS, 1))), axis=1)

            # equation (6)
            for vth, vplaylist in enumerate(all_tags):
                intersect = len(playlist_tags & vplaylist)
                weight = 1 / (pow(len(vplaylist), self.pow_alpha))
                if intersect != 0:
                    for track_i in vplaylist:
                        track_feature[track_i][vth] = intersect * weight

            # equation (7) and (8) : similarity and relevance
            for track_i in range(TOTAL_TAGS):
                feature_i = track_feature[track_i]
                if (feature_i != {}) and (not track_i in playlist_tags):

                    contain_i = self.freq_tags_powered_beta[track_i]
                    sum_of_sim = 0

                    for track_j in playlist_tags:

                        feature_j = track_feature[track_j]
                        contain_j = self.freq_tags_powered_another_beta[track_j]
                        contain = contain_i * contain_j
                        if contain == 0:
                            contain = 1.0e-10
                        sum_of_sim += (self._inner_product_feature_vector(feature_i, feature_j) / contain)

                    relevance[track_i, 1] = (1 / playlist_size) * sum_of_sim
            
            # select top 10
            relevance = relevance[relevance[:, 1].argsort()][-10:][::-1]
            pred_tags = relevance[:, 0].astype(np.int64).tolist()
            pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : [],
                "tags" : pred_tags
            })

            if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                self._auto_save(pred, auto_save_fname)

        return pd.DataFrame(pred)

    def _inner_product_feature_vector(self, v1, v2):
        '''
        v1, v2 : dictionary(key=vplaylist_id, val=features)
        '''
        result = 0
        for key, val in v1.items():
            if key in v2:
                result += (v1[key] * v2[key])
        return result
    
    def _auto_save(self, pred, auto_save_fname):
        '''
        pred : list of dictionaries
        auto_save_fname : string
        '''
        if not os.path.isdir("./_temp"):
            os.mkdir('./_temp')
        pd.DataFrame(pred).to_json(f'_temp/{auto_save_fname}.json', orient='records')

if __name__=="__main__":

    from data_util import *

    ### 1. load data
    train = pd.read_json("res/train.json")
    val = pd.read_json("res/val.json")
    # test = pd.read_json("res/test.json")

    ### 2. data preprocessing
    tag_to_id, id_to_tag = tag_id_meta(train, val)
    train = convert_tag_to_id(train, tag_to_id)
    val   = convert_tag_to_id(val  , tag_to_id)

    ### 3. modeling
    ### 3.1 hyperparameters: pow_alpha, pow_beta
    pow_alpha = 0.5
    pow_beta = 0.1

    ### 4. range setting - Neighbor.predict()
    ### 4.1 range(start, end); if end == None, then range(start, end of val)
    ### 4.2 auto_save: boolean; False(default)
    ### 4.3 return type of Neighbor.predict() : pandas.DataFrame
    pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, \
                    train=train, val=val).predict(start=0, end=None, auto_save=False)
    # print(pred)

    ### 5. data postprocessing
    pred = convert_id_to_tag(pred, id_to_tag)
    print(pred)

    ### 6. save data
    version = Neighbor.__version__
    version = version[version.find('-') + 1: version.find('.')]
    path = "submission/neighbor-tag_beta1_a5b1"
    fname = f"neighbor{version}_a{int(pow_alpha * 10)}b{int(pow_beta * 10)}"
    pred.to_json(f'{path}/{fname}.json', orient='records')