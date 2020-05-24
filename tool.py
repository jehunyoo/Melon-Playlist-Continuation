import numpy as np
import pandas as pd
import re
from tqdm.notebook import tqdm


def extract_playlist_from(data, dtype="str", dense=True, verbose=True, limit=None):
    '''
    data : train, val, test \\
    dtype : "str"(default) or "list" \\
    dense : bool; True(default) \\
    verbose : bool; True(default) \\
    limit : int or None(default); iteration limit \\
    returns : dictionaries; ko_title, ko_tag, en_title, en_tag
    '''
    if dtype == "str":
        ko = re.compile("[^ ㄱ-ㅣ가-힣]+")
        en = re.compile("[^ a-zA-Z]+")
    elif dtype == "list":
        ko = re.compile("[ㄱ-ㅣ가-힣]+")
        en = re.compile("[a-zA-Z]+")
    ko_title, ko_tag = {}, {}
    en_title, en_tag = {}, {}
    if limit:
        _range = tqdm(range(limit)) if verbose else range(limit)
    else:
        _range = tqdm(data.index) if verbose else data.index
    for i in _range:
        title = data.plylst_title[i]
        tag = " ".join(data.tags[i])
        if dtype == "str":
            a, b = ko.sub('', title).strip(), ko.sub('', tag).strip()
            c, d = en.sub('', title).strip(), en.sub('', tag).strip()
        elif dtype == "list":
            a, b = ko.findall(title), ko.findall(tag)
            c, d = en.findall(title), en.findall(tag)
        if dense:
            if a: ko_title[i] = a
            if b: ko_tag[i] = b
            if c: en_title[i] = c
            if d: en_tag[i] = d
        else:
            ko_title[i] = a
            ko_tag[i] = b
            en_title[i] = c
            en_tag[i] = d
    
    return ko_title, ko_tag, en_title, en_tag


def extract_song_from(data, dtype="str", dense=True, verbose=True, limit=None):
    '''
    data : song \\
    dtype: "str" or "list" \\
    dense : bool; True(default) \\
    verbose : bool; True(default) \\
    limit : int or None(default); iteration limit \\
    returns : dictionaries; ko_name, ko_album, ko_artist, en_name, en_album, en_artist
    '''
    if dtype == "str":
        ko = re.compile("[^ ㄱ-ㅣ가-힣]+")
        en = re.compile("[^ a-zA-Z]+")
    elif dtype == "list":
        ko = re.compile("[ㄱ-ㅣ가-힣]+")
        en = re.compile("[a-zA-Z]+")
    ko_name, ko_album, ko_artist = {}, {}, {}
    en_name, en_album, en_artist = {}, {}, {}
    if limit:
        _range = tqdm(range(limit)) if verbose else range(limit)
    else:
        _range = tqdm(data.index) if verbose else data.index
    for i in _range:
        name = data.song_name[i]
        album = data.album_name[i]
        artist = " ".join(data.artist_name_basket[i])
        if album == None: album = "" # 143209 album name == None
        if dtype == "str":
            a, b, c = ko.sub('', name).strip(), ko.sub('', album).strip(), ko.sub('', artist).strip()
            d, e, f = en.sub('', name).strip(), en.sub('', album).strip(), en.sub('', artist).strip()
        elif dtype == "list":
            a, b, c = ko.findall(name), ko.findall(album), ko.findall(artist)
            d, e, f = en.findall(name), en.findall(album), en.findall(artist)
        if dense:
            if a: ko_name[i] = a
            if b: ko_album[i] = b
            if c: ko_artist[i] = c
            if d: en_name[i] = d
            if e: en_album[i] = e
            if f: en_artist[i] = f
        else:
            ko_name[i] = a
            ko_album[i] = b
            ko_artist[i] = c
            en_name[i] = d
            en_album[i] = e
            en_artist[i] = f
    
    return ko_name, ko_album, ko_artist, en_name, en_album, en_artist


if __name__=="__main__":
    # train = pd.read_json("res/train.json")[["id", "plylst_title", "songs", "tags", "like_cnt", "updt_date"]]
    # a, b, c, d = extract_playlist_from(train, dtype="str", dense=True, limit=100)
    # print(a, b, c, d)
    song = pd.read_json("res/song_meta.json")[["id", "song_name", "artist_id_basket", "artist_name_basket",\
                                            "album_id", "album_name", "song_gn_gnr_basket",\
                                            "song_gn_dtl_gnr_basket", "issue_date"]]
    a, b, c, d, e, f = extract_song_from(song, limit=100)
    print(a, b, c, d, e, f)