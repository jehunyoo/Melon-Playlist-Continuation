import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter


class LightKNN:
    
    def __init__(self, k, rho=0.4, alpha=0.5, beta=0.5, sim="cosine", sim_normalize=False, verbose=True):
        
        self.id = None
        self.songs = None
        self.tags = None
        self.X_id = None
        self.X_songs = None
        self.X_tags = None
        self.freq = None
        
        self.k = k
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self.sim = sim
        self.sim_normalize = sim_normalize
        self.verbose = verbose
        self.__version__ = "light-1.0"

    
    def fit(self, x):
        '''
        x : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''
        self.id = x['id']
        self.songs = x['songs']
        self.tags = x['tags']
        del x
        if self.sim == "idf":
            self.freq = np.zeros(707989, dtype=np.int64)
            _playlist = tqdm(self.songs) if self.verbose else self.songs
            for _songs in _playlist:
                self.freq[_songs] += 1


    def predict(self, X, start=0, end=None, auto_save=False, auto_save_step=500, auto_save_fname='auto_save'):
        '''
        X : pandas.DataFrame; columns=['id', 'songs', 'tags']
        save_step : int
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
                S = np.zeors(len(V))
            else:
                S = np.array([self._sim(u, v) for v in V])

            if len(t) == 0 or self.beta == 0:
                T = np.zeros(len(Y))
            else:
                T = np.array([self._sim(t, w) for w in W])
            
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

                if len(songs) < 100:
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

            if (auto_save == True) and (uth + 1 % auto_save_step == 0):
                self._auto_save(pred, auto_save_fname)
        
        return pd.DataFrame(pred)
    

    def _sim(self, u, v):
        '''
        u : set (playlist in train data)
        v : set (playlist in test data)
        '''
        if self.sim == "cosine":
            if self.sim_normalize:
                try:
                    len(u & v) / ((len(u) ** 0.5) * (len(v) ** 0.5))
                except:
                    return 0
            else:
                return len(u & v)
        
        elif self.sim == "idf":
            freq = self.freq[list(u & v)]
            freq = 1 / (((freq - 1) ** self.rho) + 1) # numpy!
            if self.sim_normalize:
                try:
                    return freq.sum() / ((len(u) ** 0.5) * (len(v) ** 0.5))
                except:
                    return 0
            else:
                return freq.sum()
    
    def _auto_save(self, pred, auto_save_fname):
        
        if not os.path.isdir("./_temp"):
            os.mkdir('./_temp')
        pd.DataFrame(pred).to_json(f'_temp/{fname}.json', orient='records')


if __name__=="__main__":
    import pickle
    with open("bin/Xs.p", 'rb') as f:
        Xs = pickle.load(f)
    x = Xs[0]
    X = Xs[1]

    knn = LightKNN(100, sim='idf')
    knn.fit(x)
    pred = knn.predict(X, end=1)
    print(pred)