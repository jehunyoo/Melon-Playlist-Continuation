import numpy as np
import pandas as pd
import pickle
from warnings import warn
from tqdm.notebook import tqdm


class SimKNN:

    def __init__(self, k, rho=0.4, beta=0.7, gamma=0.3, sim=["idf", "cosine"], verbose=True):
        '''
        k : int \\
        rho : float; 0.4(default) only for idf \\
        alpha, beta, gamma : float
        sim : list of length 2; ["idf", "cosine"](default) \\
              "idf", "amplification", "cosine", "position", etc or function \\
        verbose : boolean
        '''
        self.id = None
        self.songs = None
        self.tags = None
        self.freq = None  # numpy.ndarray
        self.X = None
        # hyperparameter
        self.k = k
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        # ---
        self.sim = sim
        self.verbose = verbose
        self.__version__ = "3.0"

        self._check()


    def fit(self, x):
        '''
        x : pandas.DataFrame (columns=['id', 'songs', 'tags'])
        '''
        self.id = x['id']                   # pandas.Series of int
        self.songs = x['songs'].to_numpy()  # numpy.ndarray of list of int
        self.tags = x['tags'].to_numpy()    # numpy.ndarray of list of int
        del x
        if self.sim[0] == "idf":
            self.freq = np.zeros(707989, dtype=np.int64)
            _playlist = tqdm(self.songs) if self.verbose else self.songs
            for _songs in _playlist:
                self.freq[_songs] += 1



    
    def predict(self, X, start=0, end=None, inter=False, save_fname=None, save_interval=1000):
        '''
        parameters \\
            X : pandas.DataFrame (columns=['id', 'songs', 'tags']) \\
            start : int \\
            end : int \\
            inter : boolean; if predict songs and tags together or not
            save_fname : string
            save_interval : int
        returns \\
            pandas.DataFrame (columns=['id', 'songs', 'rel_songs', 'tags', 'rel_tags'])
    '''
        self.X_id = X['id']                   # pandas.Series of int
        self.X_songs = X['songs'].to_numpy()  # numpy.ndarray of list of int
        self.X_tags = X['tags'].to_numpy()    # numpy.ndarray of list of int
        del X

        pred = None

        if end:
            _range = tqdm(range(start, end)) if self.verbose else range(start, end)
        else:
            _range = tqdm(self.X_id.index) if self.verbose else self.X_id.index
        for uth in _range:
            k = self.k
            pred_songs = set()
            pred_tags = set()

            # interconnection check
            if inter:
                S = np.array([self._sim(uth, vth) for vth in self.id.index]) # similarities
            else:
                S_songs = np.array([self._sim(uth, vth, target='songs') for vth in self.id.index])
                S_tags = np.array([self._sim(uth, vth, target='tags') for vth in self.id.index])

            while (len(pred_songs) < 100) or (len(pred_tags) < 10):
                
                # inter check
                if not inter:
                    S = S_songs

                top = S.argsort()[-k:]  # top k indicies of v == vths
                norm = S[top].sum()

                # predict songs
                songs = np.unique(np.concatenate(self.songs[top]))
                songs = np.setdiff1d(tracks, self.X_songs[uth], assume_unique=True)

                R = np.array([( song, np.sum([S[vth] if song in self.songs[vth] else 0 for vth in top]) / norm) \
                             for song in songs]) # (id, rel)
                del songs

                R = R[R[:, 1].argsort()][::-1][:100]

                # inter check
                if not inter:
                    del S, top, norm, R
                    S = S_tags
                    top = S.argsort()[-k:]
                    norm = S[top].sum()
                
                # predict tags
                tags = np.unique(np.concatenate(self.tags[top]))
                tags = np.setdiff1d(tags, self.X_tags[uth], assume_unique=True)

                R = np.array([( tag, np.sum([S[vth] if tag in self.tags[vth] else 0 for vth in top]) / norm) \
                            for tag in tags]) # (id, rel)
                del tags

                R = R[R[:, 1].argsort()][::-1][:10]



    def _sim(self, uth, vth, target=None):
        '''
        uth : int; u is index of playlist in test.json \\
        vth : int; v is index of playlist in train.json \\
        target : string; 'songs' or 'tags
        '''
        if hasattr(self.sim, '__call__'):
            return self.sim(uth, vth)
        
        # songs
        if target == 'songs' or target == None:
            if self.X_songs[uth] == []:
                songs = 0
            elif self.sim[1] == "idf":
                u = self.X_songs[uth]
                v = self.songs[vth]
                freq = self.freq[np.intersect1d(u, v)]
                freq = 1 / (((freq - 1) ** self.rho) + 1) # numpy!
                songs = freq.sum() / ((len(u) ** 0.5) * (len(v) ** 0.5))
            elif self.sim[1] == "cosine":
                u = self.X_songs[uth]
                v = self.songs[vth]
                songs = np.intersect1d(u, v).size / ((len(u) ** 0.5) * (len(v).size ** 0.5))
            # {{ add other similarities here }}
            if target == 'songs':
                return songs

        # tags
        if target == 'tags' or target == None:
            if self.X_tags[uth] == [] or self.tags[vth] == []:
                tags = 0
            elif self.sim[2] == "idf":
                tags = None
            elif self.sim[2] == "cosine":
                u = self.X_tags[uth]
                v = self.tags[vth]
                tags = np.intersect1d(u, v).size / ((len(u) ** 0.5) * (len(v) ** 0.5))
            # {{ add other similarities here }}
            if target == 'tags':
                return tags

        return (self.beta * songs) + (self.gamma * tags)

        

    def _check(self):
        if self.beta + self.gamma != 1:            
            warn("beta + gamma == 1 is recommended.")
        if type(self.k) == type(1):
            pass
        else:
            raise TypeError(self.k)
        sim_keys = ["idf", "cosine"]
        if type(self.sim) == list:
            for sim in self.sim:
                if not (sim in sim_keys):
                    raise KeyError(sim)
        elif hasattr(self.sim, '__call__'):
            pass
        else:
            raise KeyError(self.sim)

    def _save(self, pred, save_fname):
        with open(save_fname, 'wb') as f:
            pickle.dump(pred, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            saved = pickle.load(f)
        return saved



if __name__=="__main__":
    import pickle
    with open("bin/Xs.p", 'rb') as f:
        Xs = pickle.load(f)
    simknn = SimKNN(k=200, sim=["cosine", "idf", "cosine"], beta=0.5, gamma=0.5)
    simknn.fit(x=Xs[0])
    start, end = 19, 22
    pred = simknn.predict(X=Xs[1], start=start, end=end)
    print(pred.loc[[i for i in range(start, end)], ["songs", "tags", "rel_tags"]])
    pred = simknn.predict(X=Xs[1], start=start, end=end, inter=False)
    print(pred.loc[[i for i in range(start, end)], ["songs", "tags", "rel_tags"]])


# TODO:
# 1. tags numbering (integer)
# 2. simKNNv3 마무리
# 3. simKNNv2로 numbering 한거 돌려보기 (속도 개선 해야됨)
# 4. light mode 만들기 (좀더 빠르고 가벼운 모델, 아니면 임시로 대충 만들기)