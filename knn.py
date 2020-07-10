import numpy as np
import pandas as pd
import os
from collections import Counter
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

        for uth in _range:

            playlist_songs = set(self.val_songs[uth])
            playlist_tags = set(self.val_tags[uth])
            k = self.k

            if len(playlist_songs) == 0 or self.alpha == 0:
                simSongs = np.zeros(len(all_songs))
            else:
                simSongs = np.array([self._sim(playlist_songs, vplaylist, self.sim_songs, opt="songs") for vplaylist in all_songs])

            if len(playlist_tags) == 0 or self.beta == 0:
                simTags = np.zeros(len(all_tags))
            else:
                simTags = np.array([self._sim(playlist_tags, vplaylist, self.sim_tags, opt="tags") for vplaylist in all_tags])
            
            # TODO: normalize simSongs and simTags
            sim_score = (self.alpha * simSongs) + (self.beta * simTags)

            songs = set()
            tags = []

            # TODO: add condition (len(tags) < 10)
            while len(songs) < 100:
                top = sim_score.argsort()[-k:] # top k indicies of playlists in train

                _songs = []
                _tags = []

                # for vth playlist in train
                for vth in top:
                    _songs += self.train_songs[vth]
                    _tags += self.train_tags[vth]
                songs = set(_songs) - playlist_tags

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

        if sim == "cos":
            if self.sim_normalize:
                try:
                    len(u & v) / ((len(u) ** 0.5) * (len(v) ** 0.5))
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
                    return freq.sum() / ((len(u) ** 0.5) * (len(v) ** 0.5))
                except:
                    return 0
            else:
                return freq.sum()
        
        elif sim == "jaccard":
            return len(u & v) / len(u | v)
    
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
    train = pd.read_json("res/train.json")
    val = pd.read_json("res/val.json")
    # test = pd.read_json("res/test.json")

    ### 2. modeling
    ### 2.1 hyperparameters: k, rho, alpha, beta
    ### 2.2 parameters: sim_songs, sim_tags, sim_normalize
    k = 100
    rho = 0.4
    alpha = 0.5
    beta = 0.5
    sim_songs = "idf"
    sim_tags = "cos"
    sim_normalize = False

    ### 3. range setting - KNN.predict()
    ### 3.1 range(start, end); if end == None, then range(start, end of val)
    ### 3.2 auto_save: boolean; False(default)
    ### 3.3 return type of KNN.predict() : pandas.DataFrame
    pred = KNN(k=k, rho=rho, alpha=alpha, beta=beta, \
               sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
               train=train, val=val, verbose=True, version_check=True).predict(start=0, end=10, auto_save=False)
    # print(pred)

    ### 4. save data
    version = KNN.__version__
    version = version[version.find('-') + 1: version.find('.')]
    path = "."
    fname = f"knn{version}_k{k}rho{int(rho * 10)}a{int(alpha * 10)}b{int(beta * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
    pred.to_json(f'{path}/{fname}.json', orient='records')


    # import pickle
    # with open("bin/Xs.p", 'rb') as f:
    #     Xs = pickle.load(f)
    # x = Xs[0]
    # X = Xs[1]
    # XX = Xs[2]

    # knn = KNN(100, sim_songs='cos', alpha=0.5, beta=0.5)
    # knn.fit(x)
    # for i in [2948, 3312, 3908, 5452, 5474, 18110, 18638, 21410, 22189]:
    #     pred = knn.predict(X, start=i, end=i+1)
    #     print(i, pred)