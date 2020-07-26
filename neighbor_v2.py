import numpy as np
import pandas as pd
import os
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm

class Neighbor:
    '''
    Neighbor-based Collaborative Filtering
    > Neighbor-2.0 Version Update
      + date checking
    '''

    __version__ = "Neighbor-2.1"

    def __init__(self, pow_alpha, pow_beta, train=None, val=None, song_meta=None, \
                 verbose=True, version_check=True):
        '''
        pow_alpha, pow_beta : float (0<= pow_alpha, pow_beta <= 1)
        train, val, song_meta : pandas.DataFrame
        verbose : boolean; True(default)
        version_check : boolean; True(default)
        '''

        self.train_id = train["id"].copy()
        self.train_songs = train["songs"].copy()
        self.train_tags = train["tags"].copy()
        del train

        self.val_id = val["id"].copy()
        self.val_songs = val["songs"].copy()
        self.val_tags = val["tags"].copy()
        self.val_updt_date = val["updt_date"].copy()
        del val

        self.song_meta_issue_date = song_meta["issue_date"].copy()
        del song_meta

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

        TOTAL_SONGS = 707989      # total number of songs

        freq_songs = np.zeros(TOTAL_SONGS, dtype=np.int64)
        for _songs in self.train_songs:
            freq_songs[_songs] += 1
        
        self.freq_songs_powered_beta = np.power(freq_songs, self.pow_beta)
        self.freq_songs_powered_another_beta = np.power(freq_songs, 1-self.pow_beta)

        for idx in self.val_id.index:
            self.val_updt_date.at[idx] = int(''.join(self.val_updt_date[idx].split()[0].split('-')))
        self.val_updt_date.astype(np.int64)
            

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
        all_songs = [set(songs) for songs in self.train_songs]  # list of set
        all_tags  = [set(tags)  for tags  in self.train_tags ]  # list of set

        # TODO: use variables instead of constants
        TOTAL_SONGS = 707989      # total number of songs
        MAX_SONGS_FREQ = 2175     # max freqency of songs for all playlists in train data
        TOTAL_PLAYLISTS = 115071  # total number of playlists

        for uth in _range:
            
            playlist_songs = set(self.val_songs[uth])
            playlist_tags = set(self.val_tags[uth])
            playlist_updt_date = self.val_updt_date[uth]  # type : np.int64
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
            
            # sort relevance
            relevance = relevance[relevance[:, 1].argsort()][::-1]
            sorted_songs = relevance[:, 0].astype(np.int64).tolist()

            # check if issue_date of songs is earlier than updt_date of playlist
            pred_songs = []
            for track_i in sorted_songs:
                if self.song_meta_issue_date[track_i] <= playlist_updt_date:
                    pred_songs.append(track_i)
                    if len(pred_songs) == 100:
                        break

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

    ### 1. load data
    song_meta = pd.read_json("res/song_meta.json")
    train = pd.read_json("res/train.json")
    val = pd.read_json("res/val.json")
    # test = pd.read_json("res/test.json")

    ### 2. modeling
    ### 2.1 hyperparameters: pow_alpha, pow_beta
    pow_alpha = 0.5
    pow_beta = 0.1

    ### 3. range setting - Neighbor.predict()
    ### 3.1 range(start, end); if end == None, then range(start, end of val)
    ### 3.2 auto_save: boolean; False(default)
    ### 3.3 return type of Neighbor.predict() : pandas.DataFrame
    pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, \
                    train=train, val=val, song_meta=song_meta).predict(start=0, end=None, auto_save=False)
    # print(pred)

    ### 4. save data
    version = Neighbor.__version__
    version = version[version.find('-') + 1: version.find('.')]
    path = "."
    fname = f"neighbor{version}_a{int(pow_alpha * 10)}b{int(pow_beta * 10)}"
    pred.to_json(f'{path}/{fname}.json', orient='records')