import numpy as np
import pandas as pd
import os
from tqdm import tqdm


class CFKNN:

    __version__ = "CFKNN-2.2"

    def __init__(self, k, pow_alpha, pow_beta, train=None, val=None, verbose=True):
        '''
        k : int
        pow_alpha, pow_beta : float (0<= pow_alpha, pow_beta <= 1)
        train, val : pandas.DataFrame
        verbose : boolean
        '''

        self.train_id = train["id"]
        self.train_songs = train["songs"]
        self.train_tags = train["tags"]
        del train

        self.val_id = val["id"]
        self.val_songs = val["songs"]
        self.val_tags = val["tags"]
        del val

        self.k = k
        self.pow_alpha = pow_alpha
        self.pow_beta = pow_beta

        self.verbose = verbose

        if not (0 <= self.pow_alpha <= 1):
            raise ValueError('pow_alpha is out of [0,1].')
        if not (0 <= self.pow_beta <= 1):
            raise ValueError('pow_beta is out of [0,1].')

        freq_songs = np.zeros(707989, dtype=np.int64)
        for _songs in self.train_songs:
            freq_songs[_songs] += 1
        
        self.freq_songs_powered_beta = np.power(freq_songs, self.pow_beta)
        self.freq_songs_powered_another_beta = np.power(freq_songs, 1-self.pow_beta)
        
            

    def predict(self, start=0, end=None, auto_save=False, auto_save_step=500, auto_save_fname='auto_save'):
        '''
        start, end : (start, end>0) == range(start, end), (start>0, end=None) == range(start, end of X)
                     (end = None) == all range of X
        auto_save : boolean; False(default)
        auto_save_step : int; 500(default)
        auto_save_fname : string (without extension); 'auto_save'(default)
        @returns : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''

        if end:
            _range = tqdm(range(start, end)) if self.verbose else range(start, end)
        elif start > 0 and end == None:
            _range = tqdm(range(start, self.val_id.index.stop)) if self.verbose else range(start, self.val_id.index.stop)
        else:
            _range = tqdm(self.val_id.index) if self.verbose else self.val_id.index

        pred = []
        all_songs = [set(songs) for songs in self.train_songs] # list of set
        all_tags =  [set(tags) for tags in self.train_tags]    # list of set

        # TODO: use variables instead of constants
        TOTAL_SONGS = 707989      # total number of songs
        MAX_SONGS_FREQ = 2175     # max freqency of songs for all playlists in train data
        TOTAL_PLAYLISTS = 115071  # total number of playlists

        for uth in _range:
            
            playlist_songs = set(self.val_songs[uth])
            playlist_tags = set(self.val_tags[uth])
            playlist_size = len(playlist_songs)

            if len(playlist_songs) == 0:
                pred.append({
                    "id" : int(self.val_id[uth]),
                    "songs" : [],
                    "tags" : []
                })
                if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                    self._auto_save(pred, auto_save_fname)
                continue

            track_feature = {track_i : {} for track_i in range(TOTAL_SONGS)}
            # relevance = np.zeros((TOTAL_SONGS, 2))
            relevance = np.concatenate((np.arange(TOTAL_SONGS).reshape(TOTAL_SONGS, 1), np.zeros((TOTAL_SONGS, 1))), axis=1)
            k = self.k

            # equation (6)
            for vth, vplaylist in enumerate(all_songs):
                intersect = len(playlist_songs & vplaylist)
                weight = 1 / (pow(len(vplaylist), self.pow_alpha))
                if intersect != 0:
                    for track_i in vplaylist:
                        track_feature[track_i][vth] = intersect * weight

            # equation (7) and (8) : similarity and relevance
            for track_i in range(TOTAL_SONGS):
                feature_i = track_feature[track_i]
                if (feature_i != {}) and (not track_i in playlist_songs):

                    contain_i = self.freq_songs_powered_beta[track_i]
                    sum_of_sim = 0

                    for track_j in playlist_songs:

                        feature_j = track_feature[track_j]
                        contain_j = self.freq_songs_powered_another_beta[track_j]
                        contain = contain_i * contain_j
                        if contain == 0:
                            contain = 1.0e-10
                        sum_of_sim += (self._inner_product_feature_vector(feature_i, feature_j) / contain)

                    relevance[track_i, 1] = (1 / playlist_size) * sum_of_sim
            
            # select top 100
            relevance = relevance[relevance[:, 1].argsort()][-100:][::-1]
            pred_songs = relevance[:, 0].astype(np.int64).tolist()
            pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : pred_songs,
                "tags" : []
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

    # data_load
    train = pd.read_json("res/train.json")
    val = pd.read_json("res/val.json")
    # test = pd.read_json("res/test.json")

    # modeling
    pred = CFKNN(k=100, pow_alpha=1, pow_beta=0.5, train=train, val=val).predict(end=10)
    print(pred)