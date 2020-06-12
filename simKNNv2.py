import numpy as np
import pandas as pd
import pickle
from warnings import warn
from tqdm.notebook import tqdm

class SimKNN:

    def __init__(self, k, rho=0.4, beta=0.7, gamma=0.3, \
                 sim=["cosine", "idf", "cosine"], verbose=True):
        '''
        k : int \\
        rho : float; 0.4(default) only for idf \\
        alpha, beta, gamma : float
        sim : list of length 3; ["cosine", "idf", "cosine"](default) \\
              "idf", "amplification", "cosine", "position", etc or function \\
        verbose : boolean
        '''
        self.id = None
        self.title = None
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
        self.__version__ = "2.0"

        self._check()


    def fit(self, x):
        '''
        x : pandas.DataFrame (columns=['id', 'plylst_title', 'songs', 'tags'])
        '''
        self.id = x['id']               # pandas.Series of int
        # self.title = x['plylst_title']  # pandas.Series of string
        self.songs = x['songs'].to_numpy()         # pandas.Series of list
        self.tags = x['tags'].to_numpy()           # pandas.Series of list
        del x
        if self.sim[1] == "idf":
            self.freq = np.zeros(707989, dtype=np.int64)
            _playlist = tqdm(self.songs) if self.verbose else self.songs
            for _songs in _playlist:
                self.freq[_songs] += 1


    def predict(self, X, start=0, end=None, inter=True, save_fname=None, save_interval=1000):
        '''
        parameters \\
            X : pandas.DataFrame (columns=['id', 'plylst_title', 'songs', 'tags']) \\
            start : int \\
            end : int \\
            inter : boolean; if predict songs and tags together or not
        returns \\
            pandas.DataFrame (columns=['id', 'songs', 'rel_songs', 'tags', 'rel_tags'])
        '''
        self.X_id = X['id']               # pandas.Series of int
        # self.X_title = X['plylst_title']  # pandas.Series of string
        self.X_songs = X['songs'].to_numpy()         # pandas.Series of list
        self.X_tags = X['tags'].to_numpy()           # pandas.Series of list
        del X

        pred = pd.DataFrame(index=self.X_id.index, columns=["id", "songs", "rel_songs", "tags", "rel_tags"])
        pred['id'] = pred['id'].astype('object')
        pred['songs'] = pred['songs'].astype('object')
        pred['rel_songs'] = pred['rel_songs'].astype('object')
        pred['tags'] = pred['tags'].astype('object')
        pred['rel_tags'] = pred['rel_tags'].astype('object')

        if end:
            _range = tqdm(range(start, end)) if self.verbose else range(start, end)
        else:
            _range = tqdm(self.X_id.index) if self.verbose else self.X_id.index
        for uth in _range:
            
            # interconnection check
            if inter:
                S = np.array([self._sim(uth, vth) for vth in self.id.index]) # similarities
            else:
                self.beta = 1
                S = np.array([self._sim(uth, vth, target='songs') for vth in self.id.index])
            top = S.argsort()[-self.k:]
            norm = S[top].sum()
            
            # predict songs
            tracks = np.unique(np.concatenate(self.songs[top])) # all tracks in top k playlists
            tracks = np.setdiff1d(tracks, self.X_songs[uth], assume_unique=True) # remove common songs

            if tracks.size < 100:
                print(f"{uth}(songs) {tracks.size}")

            R = np.array([( track, np.sum([S[vth] if track in self.songs[vth] else 0 for vth in top]) / norm) \
                            for track in tracks])
            del tracks

            R = R[R[:, 1].argsort()][::-1][:100]

            pred.at[uth, "id"] = self.X_id[uth]
            pred.at[uth, "songs"] = R[:, 0].astype(np.int64)
            pred.at[uth, "rel_songs"] = R[:, 1]
            del R

            # interconnection check
            if inter: pass
            else:
                del S, top, norm
                self.gamma = 1
                S = np.array([self._sim(uth, vth, target='tags') for vth in self.id.index])
                top = S.argsort()[-self.k:]
                norm = S[top].sum()
            
            # predict tags
            stickers = np.unique(np.concatenate(self.tags[top]))
            stickers = np.setdiff1d(stickers, self.X_tags[uth])
            
            if stickers.size < 10:
                print(f"{uth}(tags) {stickers.size}")
            
            R = np.array([( sticker, np.sum([S[vth] if sticker in self.tags[vth] else 0 for vth in top]) / norm) \
                            for sticker in stickers])
            del stickers
            
            try:
                R = R[R[:, 1].argsort()][::-1][:10]

                pred.at[uth, "tags"] = R[:, 0]
                pred.at[uth, "rel_tags"] = R[:, 1]
                del S, top, norm, R
            except Exception as e:
                print(f"{uth} : {e}")

            # temporary save
            if save_fname and (uth + 1) % save_interval == 0:
                self._save(pred, save_fname)

        return pred


    def _sim(self, uth, vth, target=None):
        '''
        uth : int; u is index of playlist in test.json \\
        vth : int; v is index of playlist in train.json
        '''
        if hasattr(self.sim, '__call__'):
            return self.sim(uth, vth)

        # title
        # title = 0 # FIXME
        # if self.X_title[uth] == '':
        #     title = 0

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
        else:
            songs = 0

        # tags
        if target == 'tags' or target == None:
            if self.X_tags[uth] == [] or self.tags[vth] == []:
                tags = 0
            elif self.sim[2] == "idf":
                tags = None
            elif self.sim[2] == "cosine":
                u = self.X_tags[uth] # list
                v = self.tags[vth]   # list
                tags = np.intersect1d(u, v).size / ((len(u) ** 0.5) * (len(v) ** 0.5))
            # {{ add other similarities here }}
        else:
            tags = 0

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
