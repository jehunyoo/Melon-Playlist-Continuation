import numpy as np
import pandas as pd
import os
from data_util import *
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm

class Neighbor:
    '''
    Neighbor-based Collaborative Filtering
    version Neighbor-3.0 updates
    + tag prediction
    + preprocessing for tag prediction
    '''

    __version__ = "Neighbor-3.0"


    # TODO: Remove unnecessary parameters
    def __init__(self, pow_alpha, pow_beta, train=None, val=None, song_meta=None, \
                 verbose=True, version_check=True):
        '''
        pow_alpha, pow_beta : float (0<= pow_alpha, pow_beta <= 1)
        train, val, song_meta : pandas.DataFrame
        verbose : boolean; True(default)
        version_check : boolean; True(default)
        '''
        ### 1. data sets
        ### 1.1 convert tag to tag_id
        tag_to_id, id_to_tag = tag_id_meta(train, val)
        train = convert_tag_to_id(train, tag_to_id)
        val   = convert_tag_to_id(val  , tag_to_id)
        self._id_to_tag = id_to_tag

        ### 1.2
        self.train_id = train["id"].copy()
        self.train_songs = train["songs"].copy()
        self.train_tags = train["tags"].copy()

        self.val_id = val["id"].copy()
        self.val_songs = val["songs"].copy()
        self.val_tags = val["tags"].copy()
        self.val_updt_date = val["updt_date"].copy()

        self.song_meta_issue_date = song_meta["issue_date"].copy()


        ### ?. parameters
        self.pow_alpha = pow_alpha
        self.pow_beta = pow_beta

        self.verbose = verbose
        self.__version__ = Neighbor.__version__

        if not (0 <= self.pow_alpha <= 1):
            raise ValueError('pow_alpha is out of [0,1].')
        if not (0 <= self.pow_beta <= 1):
            raise ValueError('pow_beta is out of [0,1].')

        TOTAL_SONGS = song_meta.shape[0]     # total number of songs
        TOTAL_TAGS  = len(id_to_tag.keys())  # total number of tags
        TOTAL_PLAYLISTS = train.shape[0]     # total number of playlists

        ### 2. data preprocessing
        ### 2.1 transform date format in val
        for idx in self.val_id.index:
            self.val_updt_date.at[idx] = int(''.join(self.val_updt_date[idx].split()[0].split('-')))
        self.val_updt_date.astype(np.int64)

        ### 2.2 count frequency of songs in train and compute matrices
        freq_songs = np.zeros(TOTAL_SONGS, dtype=np.int64)
        for _songs in self.train_songs:
            freq_songs[_songs] += 1
        MAX_SONGS_FREQ = np.max(freq_songs)
        self.freq_songs_powered_beta = np.power(freq_songs, self.pow_beta)
        self.freq_songs_powered_another_beta = np.power(freq_songs, 1 - self.pow_beta)

        ### 2.3 count frequency of tags in train and compute matrices
        freq_tags = np.zeros(TOTAL_TAGS, dtype=np.int64)
        for _tags in self.train_tags:
            freq_tags[_tags] += 1
        MAX_TAGS_FREQ = np.max(freq_tags)
        self.freq_tags_powered_beta = np.power(freq_tags, self.pow_beta)
        self.freq_tags_powered_another_beta = np.power(freq_tags, 1 - self.pow_beta)

        ### constants
        self.TOTAL_SONGS     = TOTAL_SONGS
        self.MAX_SONGS_FREQ  = MAX_SONGS_FREQ
        self.TOTAL_TAGS      = TOTAL_TAGS  
        self.MAX_TAGS_FREQ   = MAX_TAGS_FREQ
        self.TOTAL_PLAYLISTS = TOTAL_PLAYLISTS

        ### 3. version check TODO: Remove unnecessity.
        if version_check:
            print(f"Neighbor version: {Neighbor.__version__}")
        
        del train, val, song_meta
            

    # TODO: Remove parameters
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

        # TODO: use variables instead of constants -> DONE
        TOTAL_SONGS     = self.TOTAL_SONGS      # total number of songs
        MAX_SONGS_FREQ  = self.MAX_SONGS_FREQ   # max frequency of songs for all playlists in train
        TOTAL_TAGS      = self.TOTAL_TAGS       # total number of tags
        MAX_TAGS_FREQ   = self.MAX_TAGS_FREQ    # max frequency of tags for all playlists in train
        TOTAL_PLAYLISTS = self.TOTAL_PLAYLISTS  # total number of playlists

        for uth in _range:
            
            playlist_songs = set(self.val_songs[uth])
            playlist_tags = set(self.val_tags[uth])
            playlist_updt_date = self.val_updt_date[uth]  # type : np.int64
            playlist_size_songs = len(playlist_songs)
            playlist_size_tags = len(playlist_tags)

            pred_songs = []
            pred_tags  = []

            if playlist_size_songs == 0 and playlist_size_tags == 0:
                pred.append({
                    "id" : int(self.val_id[uth]),
                    "songs" : [],
                    "tags" : []
                })
                if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                    self._auto_save(pred, auto_save_fname)
                continue

            if playlist_size_songs != 0:

                track_feature = {track_i : {} for track_i in range(TOTAL_SONGS)}
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

                        relevance[track_i, 1] = (1 / playlist_size_songs) * sum_of_sim
                
                # sort relevance
                relevance = relevance[relevance[:, 1].argsort()][::-1]
                sorted_songs = relevance[:, 0].astype(np.int64).tolist()

                # check if issue_date of songs is earlier than updt_date of playlist
                for track_i in sorted_songs:
                    if self.song_meta_issue_date[track_i] <= playlist_updt_date:
                        pred_songs.append(track_i)
                        if len(pred_songs) == 100:
                            break

            if playlist_size_tags != 0:
                
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

                        relevance[track_i, 1] = (1 / playlist_size_tags) * sum_of_sim
                
                # select top 10
                relevance = relevance[relevance[:, 1].argsort()][-10:][::-1]
                pred_tags = relevance[:, 0].astype(np.int64).tolist()

            pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : pred_songs,
                "tags" : pred_tags
            })

            if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                self._auto_save(pred, auto_save_fname)

        ### convert tag_id to tag
        pred = pd.DataFrame(pred)
        pred = convert_id_to_tag(pred, self._id_to_tag)

        return pred

    def _inner_product_feature_vector(self, v1, v2):
        '''
        v1, v2 : dictionary(key=vplaylist_id, val=features)
        '''
        result = 0
        for key, val in v1.items():
            if key in v2:
                result += (v1[key] * v2[key])
        return result
    
    # TODO: Remove unnecessity
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
    pow_alpha = 0.7
    pow_beta = 0.0

    ### 3. parameter setting - Neighbor.predict()
    ### 3.1 range(start, end); if end == None, then range(start, end of val)
    ### 3.2 auto_save: boolean; False(default)
    ### 3.3 return type of Neighbor.predict() : pandas.DataFrame
    pred = Neighbor(pow_alpha=pow_alpha, pow_beta=pow_beta, \
                    train=train, val=val, song_meta=song_meta).predict(start=0, end=10, auto_save=False)
    # print(pred)

    ### 4. save data
    version = Neighbor.__version__
    version = version[version.find('-') + 1: version.find('.')]
    path = "."
    fname = f"neighbor{version}_a{int(pow_alpha * 10)}b{int(pow_beta * 10)}"
    pred.to_json(f'{path}/{fname}.json', orient='records')