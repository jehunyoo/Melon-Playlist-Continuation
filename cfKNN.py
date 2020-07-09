import numpy as np
import pandas as pd
from tqdm import tqdm


class CFKNN:

    __version__ = "CFKNN-1.0"

    def __init__(self, k, pow_alpha, pow_beta, train=None, val=None, verbose=True):
        '''
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

            track_feature = np.zeros((TOTAL_SONGS, MAX_SONGS_FREQ))
            track_feature_about_v = np.zeros((TOTAL_SONGS, MAX_SONGS_FREQ), dtype=np.int64)

            relevance = np.zeros(TOTAL_SONGS)
            k = self.k

            if len(playlist_songs) == 0:
                pass

            # equation (6)
            index = {i:0 for i in range(TOTAL_SONGS)}
            for vth, vplaylist in enumerate(all_songs):
                intersect = len(playlist_songs & vplaylist)
                weight = 1 / (pow(len(vplaylist), self.pow_alpha))
                if intersect != 0:
                    for track_i in vplaylist:
                        _idx = index[track_i]
                        index[track_i] += 1
                        track_feature[track_i, _idx] = intersect * weight
                        track_feature_about_v[track_i, _idx] = vth

            # equation (7) and (8) : similarity and relevance
            for track_i in range(TOTAL_SONGS):
                if track_feature[track_i, 0] != 0:

                    feature_i = np.array([])
                    contain_i = self.freq_songs_powered_beta[track_i]
                    sum_of_sim = 0

                    for track_j in playlist_songs:

                        feature_j = np.array([])
                        contain_j = self.freq_songs_powered_another_beta[track_j]
                        sum_of_sim += sum(feature_i * feature_j) / (contain_i * contain_j)

                    relevance[track_i] = (1 / playlist_size) * sum_of_sim

        
        return track_feature, track_feature_about_v



if __name__=="__main__":

    # data_load
    train = pd.read_json("res/train.json")
    val = pd.read_json("res/val.json")
    # test = pd.read_json("res/test.json")

    # modeling
    pred = CFKNN(k=100, pow_alpha=1, pow_beta=0.5, train=train, val=val).predict(end=1)
    # print(pred)
    # track_feature = pred[0]
    # for i in range(track_feature.shape[0]):
    #     if track_feature[i, 0] != 0:
    #         print(track_feature[i, :5])
    #         print(pred[1][i, :5])
    #     if i > 1000:
    #         break