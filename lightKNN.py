import numpy as np
import pandas as pd
import os
from collections import Counter
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm


class LightKNN:

    __version__ = "light-1.2"
    
    def __init__(self, k, rho=0.4, alpha=0.5, beta=0.5, \
                 sim_songs="cosine", sim_tags="cosine", \
                 sim_normalize=False, verbose=True):
        '''
        k : int
        rho : float; 0.4(default) only for idf
        alpha, beta : float; 0.5(default)
        sim_songs, sim_tags : "cosine"(default), "idf", "jaccard"
        sim_normalize : boolean; when sim == "cosine" or "idf"
        verbose : boolean
        '''
        
        self.id = None
        self.songs = None
        self.tags = None
        self.X_id = None
        self.X_songs = None
        self.X_tags = None
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
        self.__version__ = LightKNN.__version__

    
    def fit(self, x):
        '''
        x : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''
        self.id = x['id']
        self.songs = x['songs']
        self.tags = x['tags']
        del x
        if self.sim_songs == "idf":
            self.freq_songs = np.zeros(707989, dtype=np.int64)
            _playlist = tqdm(self.songs) if self.verbose else self.songs
            for _songs in _playlist:
                self.freq_songs[_songs] += 1


    def predict(self, X, start=0, end=None, auto_save=False, auto_save_step=500, auto_save_fname='auto_save'):
        '''
        X : pandas.DataFrame; columns=['id', 'songs', 'tags']
        start, end : (start, end>0) == range(start, end), (start>0, end=None) == range(start, end of X)
                     (end = None) == all range of X
        auto_save : boolean; False(default)
        auto_save_step : int; 500(default)
        auto_save_fname : string (without extension); 'auto_save'(default)
        returns : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''

        self.X_id = X['id']
        self.X_songs = X['songs']
        self.X_tags = X['tags']
        del X

        pred = []
        V = [set(songs) for songs in self.songs] # list of list
        W = [set(tags) for tags in self.tags]    # list of list

        if end:
            _range = tqdm(range(start, end)) if self.verbose else range(start, end)
        elif start > 0 and end == None:
            _range = tqdm(range(start, self.X_id.index.stop)) if self.verbose else range(start, self.X_id.index.stop)
        else:
            _range = tqdm(self.X_id.index) if self.verbose else self.X_id.index
        for uth in _range:

            u = set(self.X_songs[uth])
            t = set(self.X_tags[uth])
            k = self.k

            if len(u) == 0 or self.alpha == 0:
                S = np.zeros(len(V))
            else:
                S = np.array([self._sim(u, v, self.sim_songs, opt="songs") for v in V])

            if len(t) == 0 or self.beta == 0:
                T = np.zeros(len(W))
            else:
                T = np.array([self._sim(t, w, self.sim_tags, opt="tags") for w in W])
            
            Q = (self.alpha * S) + (self.beta * T)

            songs = set()
            tags = []

            while len(songs) < 100:
                top = Q.argsort()[-k:] # top k indicies of v == vth

                _songs = []
                _tags = []

                for vth in top:
                    _songs += self.songs[vth]
                    _tags += self.tags[vth]
                songs = set(_songs) - u

                counts = Counter(_tags).most_common(30)
                tags = [tag for tag, _ in counts if tag not in t]
                
                k += 100
            
            norm = Q[top].sum()
            if norm == 0:
                norm = 1.0e+10 # FIXME
            
            R = np.array([(song, np.sum([S[vth] if song in V[vth] else 0 for vth in top]) / norm) for song in songs])
            R = R[R[:, 1].argsort()][-100:][::-1]
            pred_songs = R[:, 0].astype(np.int64).tolist()
            pred_tags = tags[:10]

            pred.append({
                "id" : int(self.X_id[uth]),
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
        sim : string; "cosine", "idf", "jaccard" (kind of similarity)
        opt : string; "songs", "tags"
        '''

        if sim == "cosine":
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
    import pickle
    with open("bin/Xs.p", 'rb') as f:
        Xs = pickle.load(f)
    x = Xs[0]
    X = Xs[1]
    XX = Xs[2]

    knn = LightKNN(100, sim_songs='cosine', alpha=0.5, beta=0.5)
    knn.fit(x)
    for i in [2948, 3312, 3908, 5452, 5474, 18110, 18638, 21410, 22189]:
        pred = knn.predict(X, start=i, end=i+1)
        print(i, pred)