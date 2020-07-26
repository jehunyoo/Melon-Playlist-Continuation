import numpy as np
import pandas as pd
from data_util import *

class Graph:

    def __init__(self, train=None, val=None):

        tag_to_id, id_to_tag   = tag_id_meta([train, val])
        train = convert_tag_to_id(train, tag_to_id)
        val   = convert_tag_to_id(val  , tag_to_id)

        TOTAL_TAGS  = len(train_id_to_tag) +      # total number of tags
        TOTAL_PLAYLISTS = train.shape[0]  # total number of playlists

        self.train_id = train["id"].copy()
        self.train_songs = train["songs"].copy()
        self.train_tags = train["tags"].copy()

        self.val_id = val["id"].copy()
        self.val_songs = val["songs"].copy()
        self.val_tags = val["tags"].copy()

        self.graph = np.zeros((TOTAL_TAGS, TOTAL_TAGS))

    def fit(self):

        for idx in train.index:
            




    def predict(self):
        pass





if __name__=="__main__":
    
    song_meta = pd.read_json("res/song_meta.json")
    train = pd.read_json("res/train.json")
    val = pd.read_json("res/val.json")
    # test = pd.read_json("res/test.json")

    pred = Graph().predict()