import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter


class LightKNN:
    
    def __init__(self, k, rho=0.4, sim="cosine", sim_normalize=False, verbose=True):
        
        self.id = None
        self.songs = None
        self.tags = None
        self.X_id = None
        self.X_songs = None
        self.X_tags = None
        self.freq = None
        
        self.k = k
        self.rho = rho

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


    def predict(self, X, start=0, end=None):
        '''
        X : pandas.DataFrame; columns=['id', 'songs', 'tags']
        returns : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''
        self.X_id = X['id']
        self.X_songs = X['songs']
        self.X_tags = X['tags']
        del X

        pred = []
        V = [set(songs) for songs in self.songs]

        if end:
            _range = tqdm(range(start, end)) if self.verbose else range(start, end)
        else:
            _range = tqdm(self.X_id.index) if self.verbose else self.X_id.index
        for uth in _range:

            u = set(self.X_songs[uth])
            t = set(self.X_tags[uth])
            k = self.k

            S = np.array([self._sim(u, v) for v in V])

            songs = set()
            tags = []

            while len(songs) < 100:
                top = S.argsort()[-k:] # top k indicies of v == vth

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
            
            norm = S[top].sum()
            if norm == 0:
                norm = 1.0e+10
            
            R = np.array([(song, np.sum([S[vth] if song in V[vth] else 0 for vth in top]) / norm) for song in songs])
            R = R[R[:, 1].argsort()][-100:][::-1]
            pred_songs = R[:, 0].astype(np.int64).tolist()
            pred_tags = tags[:10]

            pred.append({
                "id" : int(self.X_id[uth]),
                "songs" : pred_songs,
                "tags" : pred_tags
            })
        
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