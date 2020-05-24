import numpy as np
from tqdm.notebook import tqdm

class SimKNN:

    def __init__(self, k, rho=0.4, metric="cosine", weight="uniform", verbose=True, most=100, debug=False):
        '''
        k : int \\
        rho : float; 0.3(default) only for idf \\
        metric : string; "cosine"(default), "amplification", "idf", "position", etc or function \\
        weight : string; "uniform"(default), "distance", etc or fucntion \\
        verbose : boolean
        '''
        self.data = None
        self.data_id = None
        self.X = None
        self.X_id = None
        self.freq = None
        # hyperparameters
        self.k = k
        self.rho = rho
        # ---
        self.metric = metric
        self.weight = weight # TODO
        self.verbose = verbose
        self.most = most
        self.debug = debug
        self._checker()


    def fit(self, data):
        '''
        data : pandas.DataFrame (columns=['id', 'songs'])
        '''
        self.data = data.iloc[:, 1].apply(np.array).to_numpy()
        self.data_id = data.iloc[:, 0].copy(); del data
        if self.metric == "idf":
            self.freq = np.zeros(707989, dtype=np.int64)
            _data = tqdm(self.data) if self.verbose else self.data
            for datum in _data:
                self.freq[datum] += 1
            


    def predict(self, X, generator=False, limit=None):
        '''
        X : pandas.DataFrame (columns=['id', 'songs'])
        '''
        self.X = X.iloc[:, 1].apply(np.array).to_numpy()
        self.X_id = X.iloc[:, 0].copy(); del X
        
        pred = []
        
        if type(limit) == int and limit > 0:
            _range = tqdm(range(limit)) if self.verbose else range(limit)
        else:
            _range = tqdm(range(self.X.size)) if self.verbose else range(self.X.size)
        for u in _range: # FIXME : double for loops -> numpy broadcasting?
            S = np.array([self._sim(u, v) for v in range(self.data.size)]) # sim of row for u and v
            if self.debug:
                top = S.argsort()[-(self.k+1):-1]
            else:
                top = S.argsort()[-self.k:]
            norm = S[top].sum()
            tracks = np.unique(np.concatenate(self.data[top])) # all tracks in top k playlists
            tracks = np.setdiff1d(tracks, self.X[u], assume_unique=True) # remove common songs
            r_u = np.array([( track, np.sum([S[v] if track in self.data[v] else 0 for v in top]) / norm) for track in tracks])
            # L  r_u_hat / FIXME : double for loops
            # TODO : add weight
            del S, top, norm, tracks
            
            r_u = r_u[r_u[:, 1].argsort()][::-1][:self.most]

            # yield u, r_u[:, 0].astype(np.int64), r_u[:, 1]
            r_u = (u, r_u[:, 0].astype(np.int64), r_u[:, 1])
            # tuple (playlist order, predicted songs id, relevance)
            pred.append(r_u)
        return pred


    def _sim(self, u, v):
        '''
        u : int; u is index of playlist in test.json \\
        v : int; v is index of playlist in train.json
        '''
        if self.metric == "cosine":
            u = self.X[u] # numpy array
            v = self.data[v] # numpy array
            return np.intersect1d(u, v).size / ((u.size ** 0.5) * (v.size ** 0.5))
        elif self.metric == "idf":
            u = self.X[u]
            v = self.data[v]
            freq = self.freq[np.intersect1d(u, v)]
            freq = 1 / (((freq - 1) ** self.rho) + 1)
            return freq.sum() / ((u.size ** 0.5) * (v.size ** 0.5))
        # {{ add other similarity metrics here }}
        elif hasattr(self.metric, '__call__'):
            return self.metric(u, v)

    # def score(self, data, split=0.3, limit=limit):
    #     '''
    #     data : pandas.DataFrame (columns=['id', 'songs'])
    #     '''
    #     size = int(data.shape[0] * (1 - split))
    #     X = data.iloc[size:, :] # TODO split label
    #     y = data.iloc[size:, :] # TODO
    #     data = data.iloc[size, :]
    #     self.fit(data)
    #     pred = self.predict(X, limit=limit)
    #     return self._ndcg(y, pred)

    # def _ndcg(self, y, pred):
    #     dcg = 0.0
    #     for i, r in enumerate(pred):
    #         if r in y:
    #             dcg += 1.0 / np.log(i + 2)
    #     return dcg / self._idcgs[len(y)]

    # def _idcg(self, l):
    #     return sum((1.0 / np.log(i + 2) for i in range(l)))


    # def _remove_common_songs(self, u, r_u):
    #     u = self.X[u]
    #     _most = self.most + u.size
    #     r_u = r_u[r_u[:, 1].argsort()][::-1][:_most] # select top 100 + alpha
    #     tmp = r_u[:, 0].astype(np.int64)
    #     tmp = np.setdiff1d(tmp, u, assume_unique=True)
    #     return np.array([pair for pair in r_u if pair[0] in tmp])[:self.most]
    #     # pair = (track_id, relevance) TODO : for loop -> numpy

    def _checker(self): # error checker, save your time!
        if type(self.k) == type(1):
            pass
        else:
            raise TypeError
        metric_keys = ["cosine", "amplification", "idf", "position"]
        if self.metric in metric_keys:
            pass
        elif hasattr(self.metric, '__call__'):
            pass
        else:
            raise KeyError("invalid key : {}".format(self.metric))
        # TODO: remove weight
        weight_keys = ["uniform", "distance"]
        if self.weight in weight_keys:
            pass
        elif hasattr(self.weight, '__call__'):
            pass
        else:
            raise KeyError("invalid key : {}".format(self.weight))
