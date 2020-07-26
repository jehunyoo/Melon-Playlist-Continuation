import numpy as np
import pandas as pd
from collections import Counter
from data_util import tag_id_meta


class KNN:
    '''
    K Nearest Neighbor
    '''

    __version__ = "KNN-2.0"
    
    def __init__(self, song_k, tag_k, rho=0.4, \
                 song_k_step=50, tag_k_step=10, \
                 weight_val_songs=0.5, weight_pred_songs=0.5, \
                 weight_val_tags=0.5, weight_pred_tags=0.5, \
                 sim_songs="idf", sim_tags="idf", sim_normalize=False, \
                 train=None, val=None, song_meta=None, pred=None):
        '''
        song_k, tag_k, song_k_step, tag_k_step : int
        rho : float; 0.4(default) only for idf
        weights : float
        sim_songs, sim_tags : "idf"(default), "cos"
        sim_normalize : boolean;
        '''
        ### data sets
        self.train_id    = train["id"].copy()
        self.train_songs = train["songs"].copy()
        self.train_tags  = train["tags"].copy()

        self.val_id    = val["id"].copy()
        self.val_songs = val["songs"].copy()
        self.val_tags  = val["tags"].copy()
        self.val_updt_date = val["updt_date"].copy()

        self.song_meta_issue_date = song_meta["issue_date"].copy().astype(np.int64)

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

        self.__version__ = KNN.__version__
        
        _, id_to_tag = tag_id_meta(train, val)

        TOTAL_SONGS = song_meta.shape[0]  # total number of songs
        TOTAL_TAGS  = len(id_to_tag)      # total number of tags

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


    def predict(self):
        '''
        @returns : pandas.DataFrame; columns=['id', 'songs', 'tags']
        '''

        _range = range(self.val_id.size)

        pred = []
        all_songs = [set(songs) for songs in self.train_songs] # list of set
        all_tags =  [set(tags) for tags in self.train_tags]    # list of set

        for uth in _range:

            # predict songs by tags
            if self.val_songs[uth] == [] and self.val_tags[uth] != []:
                playlist_tags_in_pred = set(self.pred_tags[uth])
                playlist_tags_in_val  = set(self.val_tags[uth])
                playlist_updt_date = self.val_updt_date[uth]
                simTags_in_pred = np.array([self._sim(playlist_tags_in_pred, vplaylist, self.sim_tags, opt='tags') for vplaylist in all_tags])
                simTags_in_val  = np.array([self._sim(playlist_tags_in_val , vplaylist, self.sim_tags, opt='tags') for vplaylist in all_tags])
                simTags = ((self.weight_pred_tags * simTags_in_pred) / (len(playlist_tags_in_pred))) + \
                          ((self.weight_val_tags * simTags_in_val) / (len(playlist_tags_in_val)))
                songs = set()

                try:
                    song_k = min(len(simTags[simTags > 0]), self.song_k)

                except:
                    song_k = self.song_k

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
            elif self.val_songs[uth] != [] and self.val_tags[uth] == []:
                playlist_songs_in_pred = set(self.pred_songs[uth])
                playlist_songs_in_val  = set(self.val_songs[uth])
                simSongs_in_pred = np.array([self._sim(playlist_songs_in_pred, vplaylist, self.sim_songs, opt='songs') for vplaylist in all_songs])
                simSongs_in_val  = np.array([self._sim(playlist_songs_in_val , vplaylist, self.sim_songs, opt='songs') for vplaylist in all_songs])
                simSongs = ((self.weight_pred_songs * simSongs_in_pred) / (len(playlist_songs_in_pred))) + \
                           ((self.weight_val_songs * simSongs_in_val)  /  (len(playlist_songs_in_val)))
                tags = []

                try:
                    tag_k = min(len(simSongs[simSongs > 0]), self.tag_k)

                except:
                    tag_k = self.tag_k

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
        
        return pd.DataFrame(pred)
    

    def _sim(self, u, v, sim, opt):
        '''
        u : set (playlist in train data)
        v : set (playlist in test data)
        sim : string; "cos", "idf"
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

if __name__=="__main__":
    pass