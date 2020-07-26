'''
NumbaCFKNN uses Numba and it's expected to be faster than CFKNN.
Since Numba cannot interpret python dictionary, tqdm and Pandas,
there's no such types in methods called by NumbaCFKNN.predict and NumbaCFKNN.predict.
Only Pandas is declared in NumbaCFKNN.__init__.
'''


__version__ = "numba_CFKNN-1.0"


import numpy as np
import pandas as pd
import numba
import os
from tqdm import tqdm


def _data_split(train, val):
    '''
    '''

    TOTAL_PLAYLISTS_IN_TRAIN = 115071  # total number of playlists in train
    MAX_SONGS_IN_TRAIN = 200           # max number of songs in one playlists
    TOTAL_PLAYLISTS_IN_VAL = 23015     # total number of playlists in val
    MAX_SONGS_IN_VAL = 100             # max number of songs in one playlists
    # TOTAL_PLAYLISTS_IN_TEST = 10740    # total number of playlists in test
    # MAX_SONGS_IN_TEST = 100            # max number of songs in one playlists
    
    train_songs = [[-1 for _ in range(MAX_SONGS_IN_TRAIN)] for _ in range(TOTAL_PLAYLISTS_IN_TRAIN)]
    for idx1 in train.index:
        for idx2, song in enumerate(train.songs[idx1]):
            train_songs[idx1][idx2] = song
    train_songs = np.array(train_songs, dtype=np.int64)

    train_id = train["id"].to_numpy()
    # train_tags = train["tags"].to_list()
    del train

    val_songs = [[-1 for _ in range(MAX_SONGS_IN_VAL)] for _ in range(TOTAL_PLAYLISTS_IN_VAL)]
    for idx1 in val.index:
        for idx2, song in enumerate(val["songs"][idx1]):
            val_songs[idx1][idx2] = song
    val_songs = np.array(val_songs, dtype=np.int64)

    val_id = val["id"].to_numpy()
    # val_tags = val["tags"].to_list()
    del val

    return train_id, train_songs, val_id, val_songs


def predict(train, val, alpha, beta, start=0, end=None, verbose=True, \
            auto_save=False, auto_save_step=500, auto_save_fname='auto_save'):
    '''
    '''

    train_id, train_songs, val_id, val_songs = _data_split(train, val)

    freq_songs = np.zeros(707989, dtype=np.int64)
    for _songs in train_songs:
        freq_songs[_songs] += 1       
    freq_songs_powered_beta = np.power(freq_songs, beta)
    freq_songs_powered_another_beta = np.power(freq_songs, 1 - beta)

    if end:
        _range = tqdm(range(start, end)) if verbose else range(start, end)
    elif end == None:
        _range = tqdm(range(start, val_id.shape[0])) if verbose else range(start, val_id.shape[0])
    
    pred = []

    # TODO: use variables instead of constants
    TOTAL_SONGS = 707989      # total number of songs
    MAX_SONGS_FREQ = 2175     # max freqency of songs for all playlists in train data
    TOTAL_PLAYLISTS = 115071  # total number of playlists

    for uth in _range:

        playlist_songs = val_songs[uth][np.where(val_songs[uth] >= 0)].tolist()
        print(playlist_songs)

        if len(playlist_songs) == 0:
            pred.append({
                "id" : int(val_id[uth]),
                "songs" : [],
                "tags" : []
            })
            if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                _auto_save(pred, auto_save_fname)
            continue

        # track_feature = np.zeros((TOTAL_SONGS, MAX_SONGS_FREQ)) - 1                          # all values are -1 < 0
        track_feature_about_v = np.zeros((TOTAL_SONGS, MAX_SONGS_FREQ), dtype=np.int64) - 1  # all values are -1 < 0
        relevance = np.concatenate((np.arange(TOTAL_SONGS).reshape(TOTAL_SONGS, 1), np.zeros((TOTAL_SONGS, 1))), axis=1)
        index = np.zeros(TOTAL_SONGS, dtype=np.int64)

        # relevance = _calculate(train_songs, playlist_songs, \
        #                        relevance, freq_songs_powered_beta, freq_songs_powered_another_beta, \
        #                        index, alpha, beta, TOTAL_SONGS, MAX_SONGS_FREQ)

        relevance = relevance[relevance[:, 1].argsort()][-100:][::-1]
        pred_songs = relevance[:, 0].astype(np.int64).tolist()
        pred.append({
            "id" : int(val_id[uth]),
            "songs" : pred_songs,
            "tags" : []
        })

        if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
            _auto_save(pred, auto_save_fname)

    return pd.DataFrame(pred)


@numba.jit(nopython=True)
def _calculate(train_songs, playlist_songs, \
               relevance, freq_songs_powered_beta, freq_songs_powered_another_beta, \
               index, alpha, beta, TOTAL_SONGS, MAX_SONGS_FREQ):
    '''
    train_songs : numpy.ndarray (shape=(115017, 200)=(TOTAL_PLAYLISTS, MAX_SONGS_IN_TRAIN = 200))
    playlist_songs : list
    relevance : numpy.ndarray (shape=(TOTAL_SONGS, 2));
                relevance[0, :] = np.arange(TOTAL_SONGS) / relevance[1, :] = np.zeros(TOTAL_SONGS)
    '''
    vth = 0
    playlist_size = len(playlist_songs)
    track_feature = [[-1 for _ in range(MAX_SONGS_FREQ)] for _ in range(TOTAL_SONGS)]

    for _vplaylist in train_songs:
        vplaylist = _vplaylist[np.where(_vplaylist >= 0)] # numpy.ndarray
        intersect = len(set(playlist_songs) & set(vplaylist))
        if intersect == 0:
            continue
        weight = 1 / pow(vplaylist.size, alpha)
        feature_value = intersect * weight
        for track_i in vplaylist:
            _idx = index[track_i]
            index[track_i] += 1
            wow[track_i][_idx] = feature_value
        vth += 1

    _range = [track_i for track_i in range(TOTAL_SONGS)]
    for track_j in playlist_songs:
        _range.remove(track_j)

    for track_i in _range:
        feature_i = track_feature[track_i]
        # feature_i_about_v = track_feature_about_v[track_i]
        
        if (track_feature[track_i][0] != 0.0):# and (not track_i in playlist_songs):

            contain_i = freq_songs_powered_beta[track_i]
            sum_of_sim = 0

            for track_j in playlist_songs:

                feature_j = track_feature[track_j]
                # feature_j_about_v = track_feature_about_v[track_j]
                
                contain_j = freq_songs_powered_beta[track_j]
                contain = contain_i * contain_j
                if contain == 0:
                    contain = 1.0e-10
                
                # same as _feature_product(feature_i, feature_j, feature_i_about_v, feature_j_about_v)
                
                # where1 = [0, 1, 2, 3, 4]
                # where2 = [0, 1, 2, 3, 4]
                # feature_product = np.sum(feature_i[where1] * feature_j[where2])
                feature_product = 0

                sum_of_sim += (feature_product / contain)
            
            relevance[track_i, 1] = (1 / playlist_size) * sum_of_sim

    return relevance



def _auto_save(pred, auto_save_fname):
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
    alpha = 1
    beta = 0.5
    pred = predict(train, val, alpha, beta, start=100, end=120)
    print(pred)
    