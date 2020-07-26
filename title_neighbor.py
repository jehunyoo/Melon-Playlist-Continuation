import numpy as np
import pandas as pd
import os
from collections import Counter
from tool import *
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm


class Neighbor_title:

    __version__ = "Neighbor_title-1.0"
    
    def __init__(self, gamma=0.5, train=None, val=None, verbose=True, version_check=True):
        '''
        k : int
        rho : float; 0.4(default) only for idf
        alpha, beta : float; 0.5(default)
        sim_songs, sim_tags : "cos"(default), "idf", "jaccard"
        sim_normalize : boolean; when sim == "cos" or "idf"
        verbose : boolean
        '''
        self.train_title_vector, self.val_title_vector = self._title_vector(train, val)

        self.train_id = train["id"]
        self.train_songs = train["songs"]
        self.train_tags = train["tags"]
        del train

        self.val_id = val["id"]
        self.val_songs = val["songs"]
        self.val_tags = val["tags"]
        del val
        
        self.gamma = gamma

        self.verbose = verbose
        self.__version__ = Neighbor_title.__version__

        if version_check:
            print(f"KNN version: {Neighbor_title.__version__}")


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

        TOTAL_SONGS = 707989
        TOTAL_PLAYLISTS = 115071
        
        pred = []
        all_songs = [set(songs) for songs in self.train_songs] # list of set
        all_tags =  [set(tags) for tags in self.train_tags]    # list of set
        all_title = [set(title) for title in self.train_title_vector]

        for uth in _range:

            playlist_title = set(self.val_title_vector[uth])
            playlist_songs = set(self.val_songs[uth])
            playlist_tags = set(self.val_tags[uth])
            playlist_size = len(playlist_title)

            if playlist_size == 0 or len(playlist_songs) != 0 or len(playlist_tags) != 0:
                pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : [],
                "tags" : []
                })
                continue

            relevance = np.concatenate((np.arange(TOTAL_SONGS).reshape(TOTAL_SONGS, 1), np.zeros((TOTAL_SONGS, 1))), axis=1)

            for track_i in tqdm(range(TOTAL_SONGS)):
                sum_of_sim = 0

                for title_gram in playlist_title:
                    candidate1 = []
                    candidate2 = []
                    for idx in range(TOTAL_PLAYLISTS):
                        if (title_gram in all_title[idx]) and (track_i in all_songs[idx]):
                            candidate1.append(idx)
                        if track_i in all_songs[idx]:
                            candidate2.append(idx)
                    try:
                        sum_of_sim += len(candidate1) / pow(len(candidate2), self.gamma)
                    except ZeroDivisionError:
                        sum_of_sim += 0

                relevance[track_i] = sum_of_sim / playlist_size

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


    def _title_vector(self, train, val):
        
        train_ko_title, _, _, _ = extract_playlist_from(train, verbose=False)
        _, _, train_en_title, _ = extract_playlist_from(train, dtype="list", verbose=False)
        val_ko_title, _, _, _ = extract_playlist_from(val, verbose=False)
        _, _, val_en_title, _ = extract_playlist_from(val, dtype="list", verbose=False)

        data = [train_ko_title, val_ko_title]
        for dct in data:
            for key, title in dct.items():
                new_title = ''
                for word in title:
                    if word != ' ':
                        new_title += word
                dct[key] = new_title

        data = [train_ko_title, val_ko_title]
        for dct in data:
            for key, title in dct.items():
                length = len(title)
                bag = []
                for n in range(2, length):
                    bag += n_gram_sentence(n, title)
                dct[key] = bag

        TOTAL_PLAYLISTS_TRAIN = train.shape[0]
        TOTAL_PLAYLISTS_VAL   = val.shape[0]

        train_title_vector = [None for _ in range(TOTAL_PLAYLISTS_TRAIN)]
        val_title_vector   = [None for _ in range(TOTAL_PLAYLISTS_VAL)]

        for i in train.index:
            ko_vector = []
            en_vector = []
            if i in train_ko_title:
                ko_vector = train_ko_title[i]
            if i in train_en_title:
                en_vector = train_en_title[i]
            train_title_vector[i] = ko_vector + en_vector

        for i in val.index:
            ko_vector = []
            en_vector = []
            if i in val_ko_title:
                ko_vector = val_ko_title[i]
            if i in val_en_title:
                en_vector = val_en_title[i]
            val_title_vector[i] = ko_vector + en_vector
        
        return train_title_vector, val_title_vector

    
    def _auto_save(self, pred, auto_save_fname):
        '''
        pred : list of dictionaries
        auto_save_fname : string
        '''
        
        if not os.path.isdir("./_temp"):
            os.mkdir('./_temp')
        pd.DataFrame(pred).to_json(f'_temp/{auto_save_fname}.json', orient='records')


if __name__=="__main__":
    
    train = pd.read_json('res/train.json')
    val = pd.read_json('res/val.json')

    pred = Neighbor_title(gamma=0.5, train=train, val=val).predict(start=0, end=10)
    print(pred)
