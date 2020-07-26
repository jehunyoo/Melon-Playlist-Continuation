import numpy as np
import pandas as pd
import os
from collections import Counter
from data_util import tag_id_meta
from warnings import warn

warn("Unsupported module 'tqdm' is used.")
from tqdm import tqdm


class NeighborKNN:
    '''
    K Nearest Neighbor
    '''

    __version__ = "NeighborKNN-2.0"
    
    def __init__(self, song_k, tag_k, rho=0.4, \
                 song_k_step=50, tag_k_step=10, \
                 weight_val_songs=0.5, weight_pred_songs=0.5, \
                 weight_val_tags=0.5, weight_pred_tags=0.5, \
                 sim_songs="idf", sim_tags="cos", sim_normalize=False, \
                 train=None, val=None, song_meta=None, pred=None, \
                 verbose=True, version_check=True):
        '''
        k : int
        rho : float; 0.4(default) only for idf
        alpha, beta : float; 0.5(default)
        sim_songs, sim_tags : "cos"(default), "idf", "jaccard"
        sim_normalize : boolean; when sim == "cos" or "idf"
        verbose : boolean
        '''
        ### data sets
        self.train_id    = train["id"].copy()
        self.train_songs = train["songs"].copy()
        self.train_tags  = train["tags"].copy()

        self.val_id    = val["id"].copy()
        self.val_songs = val["songs"].copy()
        self.val_tags  = val["tags"].copy()
        self.val_updt_date = val["updt_date"].copy()

        self.song_meta_issue_date = song_meta["issue_date"].copy()

        self.pred_songs = pred["songs"].copy()
        self.pred_tags  = pred["tags"].copy()

        self.freq_songs = None
        self.freq_tags  = None
        
        self.song_k = song_k
        self.tag_k  = tag_k
        self.song_k_step = song_k_step
        self.tag_k_step  = tag_k_step
        self.rho = rho
        self.weight_val_songs  = weight_val_songs
        self.weight_pred_songs = weight_pred_songs
        self.weight_val_tags   = weight_val_tags
        self.weight_pred_tags  = weight_pred_tags

        self.sim_songs     = sim_songs
        self.sim_tags      = sim_tags
        self.sim_normalize = sim_normalize

        self.verbose = verbose
        self.__version__ = NeighborKNN.__version__

        if version_check:
            print(f"NeighborKNN version: {NeighborKNN.__version__}")
        
        _, id_to_tag = tag_id_meta([train, val])

        TOTAL_SONGS = song_meta.shape[0]     # total number of songs
        TOTAL_TAGS  = len(id_to_tag)  # total number of tags

        ### transform date format in val
        for idx in self.val_id.index:
            self.val_updt_date.at[idx] = int(''.join(self.val_updt_date[idx].split()[0].split('-')))
        self.val_updt_date.astype(np.int64)


        if self.sim_songs == "idf":

            self.freq_songs = np.zeros(TOTAL_SONGS, dtype=np.int64)
            for _songs in self.train_songs:
                self.freq_songs[_songs] += 1

        if self.sim_tags == "idf":

            self.freq_tags = np.zeros(TOTAL_TAGS, dtype=np.int64)
            for _tags in self.train_tags:
                self.freq_tags[_tags] += 1

        del train, val, song_meta, pred


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

            song_k = self.song_k
            tag_k  = self.tag_k

            # predict songs by tags
            if self.pred_songs[uth] == [] and self.pred_tags[uth] != []:
                playlist_tags_in_pred = set(self.pred_tags[uth])
                playlist_tags_in_val  = set(self.val_tags[uth])
                playlist_updt_date = self.val_updt_date[uth]
                simTags_in_pred = np.array([self._sim(playlist_tags_in_pred, vplaylist, self.sim_tags, opt='tags') for vplaylist in all_tags])
                simTags_in_val  = np.array([self._sim(playlist_tags_in_val , vplaylist, self.sim_tags, opt='tags') for vplaylist in all_tags])
                simTags = ((self.weight_pred_tags * simTags_in_pred) / (len(playlist_tags_in_pred))) + \
                          ((self.weight_val_tags * simTags_in_val) / (len(playlist_tags_in_val)))
                songs = set()

                while len(songs) < 100:
                    top = simTags.argsort()[-song_k:]
                    _songs = []

                    for vth in top:
                        _songs += self.train_songs[vth]
                    songs = set(_songs)

                    # check if issue_date of songs is earlier than updt_date of playlist
                    date_checked = []
                    for track_i in songs:
                        if self.song_meta_issue_date[track_i] <= playlist_updt_date:
                            date_checked.append(track_i)
                    songs = set(date_checked)

                    song_k += self.song_k_step
                
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

            # predict tags using songs
            elif self.pred_songs[uth] != [] and self.pred_tags[uth] == []:
                playlist_songs_in_pred = set(self.pred_songs[uth])
                playlist_songs_in_val  = set(self.val_songs[uth])
                simSongs_in_pred = np.array([self._sim(playlist_songs_in_pred, vplaylist, self.sim_songs, opt='songs') for vplaylist in all_songs])
                simSongs_in_val  = np.array([self._sim(playlist_songs_in_val , vplaylist, self.sim_songs, opt='songs') for vplaylist in all_songs])
                simSongs = ((self.weight_pred_songs * simSongs_in_pred) / (len(playlist_songs_in_pred))) + \
                           ((self.weight_val_songs * simSongs_in_val)  /  (len(playlist_songs_in_val)))
                tags = []
                
                while len(tags) < 10:
                    top = simSongs.argsort()[-tag_k:]
                    _tags = []
                    
                    for vth in top:
                        _tags += self.train_tags[vth]

                    counts = Counter(_tags).most_common(30)
                    tags = [tag for tag, _ in counts]

                    tag_k += self.tag_k_step
                
                pred_tags = tags[:10]

                pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : self.pred_songs[uth],
                "tags" : pred_tags
                })

            # if val.songs[uth] == [] and val.tags[uth] == [] -> pred.songs[uth] == [] and pred.tags[uth] == []
            # if val.songs[uth] != [] and val.tags[uth] != [] -> pred.songs[uth] != [] and pred.tags[uth] != []
            else:
                pred.append({
                "id" : int(self.val_id[uth]),
                "songs" : self.pred_songs[uth],
                "tags" : self.pred_tags[uth]
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
                    return len(u & v) / ((len(u) ** 0.5) * (len(v) ** 0.5))
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

    from data_util import *

    song_meta = pd.read_json("res/song_meta.json")
    train = pd.read_json("res/train.json")
    val   = pd.read_json("res/val.json")
    pred = pd.read_json("submission/neighbor3_a7b0/only_songs_from_neighbor3_a7b0.json", orient='records')

    tag_to_id, id_to_tag = tag_id_meta([train, val])
    train = convert_tag_to_id(train, tag_to_id)
    val   = convert_tag_to_id(val  , tag_to_id)
    pred  = convert_tag_to_id(pred, tag_to_id)
    
    ### 4. modeling : NeighborKNN
    ### 4.1 hyperparameters: k, rho, weights
    ### 4.2 parameters: sim_songs, sim_tags, sim_normalize
    song_k = 1
    tag_k  = 50
    song_k_step = 1
    tag_k_step  = 25
    rho = 0.4
    weight_val_songs  = 0.9
    weight_pred_songs = 1 - weight_val_songs
    weight_val_tags   = 1.0
    weight_pred_tags  = 1 - weight_val_tags
    sim_songs = "idf"
    sim_tags = "idf"
    sim_normalize = False
    pred = NeighborKNN(song_k=song_k, tag_k=tag_k, rho=rho, \
                   song_k_step=song_k_step, tag_k_step=tag_k_step, \
                   weight_val_songs=weight_val_songs, weight_pred_songs=weight_pred_songs, \
                   weight_val_tags=weight_val_tags, weight_pred_tags=weight_pred_tags, \
                   sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
                   train=train, val=val, song_meta=song_meta, pred=pred).predict(start=6000, end=11500, auto_save=True)
    pred = convert_id_to_tag(pred, id_to_tag)
    version = NeighborKNN.__version__
    version = version[version.find('-') + 1: version.find('.')]
    path = "."
    fname2 = f"neighbor-knn{version}_k{song_k}-{tag_k}step{song_k_step}-{tag_k_step}rho{int(rho * 10)}s{int(weight_val_songs * 10)}t{int(weight_val_tags * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
    pred.to_json(f'{path}/{fname2}.json', orient='records')

    # ### 4.3 run NeighborKNN.predict() : returns pandas.DataFrame
    # pred = NeighborKNN(song_k=song_k, tag_k=tag_k, rho=rho, \
    #                    song_k_step=song_k_step, tag_k_step=tag_k_step, \
    #                    weight_val_songs=weight_val_songs, weight_pred_songs=weight_pred_songs, \
    #                    weight_val_tags=weight_val_tags, weight_pred_tags=weight_pred_tags, \
    #                    sim_songs=sim_songs, sim_tags=sim_tags, sim_normalize=sim_normalize, \
    #                    train=train, val=val, song_meta=song_meta, pred=pred).predict(start=0, end=None, auto_save=True)
    # pred = convert_id_to_tag(pred, id_to_tag)
    # # print(pred)

    # ### ==============================(save data)==============================
    # version = NeighborKNN.__version__
    # version = version[version.find('-') + 1: version.find('.')]
    # path = "."
    # fname2 = f"neighbor-knn{version}_k{song_k}-{tag_k}step{song_k_step}-{tag_k_step}rho{int(rho * 10)}s{int(weight_val_songs * 10)}t{int(weight_val_tags * 10)}_{sim_songs}{sim_tags}{sim_normalize}"
    # pred.to_json(f'{path}/{fname2}.json', orient='records')
    # ### ======================================================================