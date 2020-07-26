import numpy as np
import pandas as pd
import os
from collections import Counter
from tool import *
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm


class KNN:

    __version__ = "KNN-1.0"
    
    def __init__(self, k, rho=0.4, alpha=0.5, beta=0.5, \
                 sim_songs="cos", sim_tags="cos", sim_normalize=False, \
                 train=None, val=None, verbose=True, version_check=True):
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

        self.freq_songs = None
        self.freq_tags = None
        
        self.k = k
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self.sim_songs = sim_songs
        self.sim_tags = sim_tags
        self.sim_normalize = sim_normalize

        self.verbose = verbose
        self.__version__ = KNN.__version__

        if version_check:
            print(f"KNN version: {KNN.__version__}")

        TOTAL_SONGS = 707989      # total number of songs

        if self.sim_songs == "idf":

            self.freq_songs = np.zeros(TOTAL_SONGS, dtype=np.int64)
            _playlist = self.train_songs
            for _songs in _playlist:
                self.freq_songs[_songs] += 1


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
        all_songs = [set(songs) for songs in self.train_songs] # list of set
        all_tags =  [set(tags) for tags in self.train_tags]    # list of set
        all_title = [set(title) for title in self.train_title_vector]

        for uth in _range:

            playlist_title = set(self.val_title_vector[uth])
            playlist_songs = set(self.val_songs[uth])
            playlist_tags = set(self.val_tags[uth])
            k = self.k

            if playlist_title == [] or len(playlist_songs) != 0 or len(playlist_tags) != 0:
                 pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : [],
                "tags" : []]
                })

            simTitle = np.array(len(playlist_title & vtitle_vector) for vtitle_vector in all_title])

            songs = set()
            tags = []

            # TODO: add condition (len(tags) < 10)
            while (len(songs) < 100 or len(tags) < 10):
                top = sim_score.argsort()[-k:] # top k indicies of playlists in train

                _songs = []
                _tags = []

                # for vth playlist in train
                for vth in top:
                    _songs += self.train_songs[vth]
                    _tags += self.train_tags[vth]
                songs = set(_songs) - playlist_songs

                counts = Counter(_tags).most_common(30)
                tags = [tag for tag, _ in counts if tag not in playlist_tags]
                
                k += 100
            
            norm = sim_score[top].sum()
            if norm == 0:
                norm = 1.0e+10 # FIXME
            
            relevance = np.array([(song, np.sum([simSongs[vth] if song in all_songs[vth] else 0 for vth in top]) / norm) for song in songs])
            relevance = relevance[relevance[:, 1].argsort()][-100:][::-1]
            pred_songs = relevance[:, 0].astype(np.int64).tolist()
            pred_tags = tags[:10]

            pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : pred_songs,
                "tags" : pred_tags
            })

            if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                self._auto_save(pred, auto_save_fname)
        
        return pd.DataFrame(pred)
    

    def _sim(self, u, v, sim, opt):
        '''
        u : set (playlist in train data)
        v : set (playlist in test data)
        sim : string; "cos", "idf", "jaccard" (kind of similarity)
        opt : string; "songs", "tags"
        '''

        norm = ((len(u) ** 0.5) * (len(v) ** 0.5))
        if sim == "cos":
            if self.sim_normalize:
                try:
                    return len(u & v) / norm if norm != 0 else len(u & v)
                except:
                    return 0
            else:
                return len(u & v)
        
        elif sim == "idf":
            if opt == "songs":
                freq = self.freq_songs
            elif opt == "tags":
                freq = self.freq_tags
            freq = freq[list(u & v)]
            freq = 1 / (((freq - 1) ** self.rho) + 1) # numpy!
            if self.sim_normalize:
                try:
                    return freq.sum() / norm if norm != 0 else freq.sum()
                except:
                    return 0
            else:
                return freq.sum()
        
        elif sim == "jaccard":
            return len(u & v) / len(u | v)

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
    pass