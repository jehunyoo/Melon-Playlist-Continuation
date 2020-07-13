import numpy as np
import pandas as pd
import os
from collections import Counter
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm


class NeighborKNN_song:

    __version__ = "NeighborKNN_song-1.0"
    
    def __init__(self, k, rho=0.4, weight_val=0.5, weight_pred=0.5, \
                 sim_tags="cos", sim_normalize=False, \
                 train=None, val=None, \
                 pred_tags=None, \
                 verbose=True, version_check=True):
        '''
        k : int
        rho : float; 0.4(default) only for idf
        alpha, beta : float; 0.5(default)
        sim_songs, sim_tags : "cos"(default), "idf", "jaccard"
        sim_normalize : boolean; when sim == "cos" or "idf"
        verbose : boolean
        '''
        self.train_id = train["id"].copy()
        self.train_songs = train["songs"].copy()
        self.train_tags = train["tags"].copy()
        del train

        self.val_id = val["id"].copy()
        self.val_songs = val["songs"].copy()
        self.val_tags = val["tags"].copy()
        del val

        self.pred_tags = pred_tags["tags"].copy()

        self.freq_tags = None
        
        self.k = k
        self.rho = rho
        self.weight_val = weight_val
        self.weight_pred = weight_pred

        self.sim_tags = sim_tags
        self.sim_normalize = sim_normalize

        self.verbose = verbose
        self.__version__ = NeighborKNN_song.__version__

        if version_check:
            print(f"KNN version: {NeighborKNN_song.__version__}")

        # TOTAL_TAGS = 707989  #TODO # total number of songs

        # if self.sim_tags == "idf":

        #     self.freq_tags = np.zeros(TOTAL_TAGS, dtype=np.int64)
        #     _playlist = self.train_tags
        #     for _tags in _playlist:
        #         self.freq_songs[_tags] += 1


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

            k = self.k
            
            # predict songs using tags
            if self.val_songs[uth] == [] and self.val_tags[uth] != []:
                playlist_tags_in_pred = set(self.pred_tags[uth])
                playlist_tags_in_val  = set(self.val_tags[uth])
                simTags_in_pred = np.array([self._sim(playlist_tags_in_pred, vplaylist, self.sim_tags) for vplaylist in all_tags])
                simTags_in_val  = np.array([self._sim(playlist_tags_in_val , vplaylist, self.sim_tags) for vplaylist in all_tags])
                simTags = ((self.weight_pred * simTags_in_pred) / (len(playlist_tags_in_pred))) + \
                          ((self.weight_val * simTags_in_val) / (len(playlist_tags_in_val)))
                songs = set()

                while len(songs) < 100:
                    top = simTags.argsort()[-k:]
                    _songs = []

                    for vth in top:
                        _songs += self.train_songs[vth]
                    songs = set(_songs)

                    k += 100
                
                norm = simTags[top].sum()
                if norm == 0:
                    norm = 1.0e+10 # FIXME
            
                relevance = np.array([(song, np.sum([simTags[vth] if song in all_songs[vth] else 0 for vth in top]) / norm) for song in songs])
                relevance = relevance[relevance[:, 1].argsort()][-100:][::-1]
                pred_songs = relevance[:, 0].astype(np.int64).tolist()

                pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : pred_songs,
                "tags" : self.pred_tags[uth]
                })
            
            else:
                pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : [],
                "tags" : []
                })

            if (auto_save == True) and ((uth + 1) % auto_save_step == 0):
                self._auto_save(pred, auto_save_fname)
        
        return pd.DataFrame(pred)
    

    def _sim(self, u, v, sim):
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
    pred_tags = pd.read_json("submission/neighbor-tag_beta1_a7b0/neighbortag_beta1_a7b0.json", orient='records')
    # test = pd.read_json("res/test.json")

    ### 2. modeling
    ### 2.1 hyperparameters: k, rho, alpha, beta
    ### 2.2 parameters: sim_songs, sim_tags, sim_normalize
    k = 100
    rho = 0.4
    weight_val = 0.5
    weight_pred = 0.5
    sim_tags = "cos"
    sim_normalize = False

    ### 3. range setting - KNN.predict()
    ### 3.1 range(start, end); if end == None, then range(start, end of val)
    ### 3.2 auto_save: boolean; False(default)
    ### 3.3 return type of KNN.predict() : pandas.DataFrame
    pred = NeighborKNN_song(k=k, rho=rho, weight_val=weight_val, weight_pred=weight_pred, \
               sim_tags=sim_tags, sim_normalize=sim_normalize, \
               train=train, val=val, pred_tags=pred_tags, verbose=True, version_check=True).predict(start=0, end=None, auto_save=False)
    print(pred)

    ### 4. save data
    version = NeighborKNN_song.__version__
    version = version[version.find('-') + 1: version.find('.')]
    path = "."
    fname = f"neighbor-knn-song{version}_k{k}rho{int(rho * 10)}v{int(weight_val * 10)}p{int(weight_pred * 10)}_{sim_tags}{sim_normalize}"
    pred.to_json(f'{path}/{fname}.json', orient='records')