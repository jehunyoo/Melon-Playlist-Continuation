import numpy as np
import pandas as pd
import re
from tqdm.notebook import tqdm


def extract_playlist_from(data, dtype="str", dense=True, verbose=True, limit=None):
    '''
    parameters \\
        data : train, val, test \\
        dtype : "str"(default) or "list" \\
        dense : bool; True(default) \\
        verbose : bool; True(default) \\
        limit : int or None(default); iteration limit \\
    returns \\
        dictionaries; ko_title, ko_tag, en_title, en_tag
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
            a, b = ko.sub(' ', title).strip(), ko.sub(' ', tag).strip()
            c, d = en.sub(' ', title).strip(), en.sub(' ', tag).strip()
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
    parameters \\
        data : song \\
        dtype: "str" or "list" \\
        dense : bool; True(default) \\
        verbose : bool; True(default) \\
        limit : int or None(default); iteration limit \\
    returns \\
        dictionaries; ko_name, ko_album, ko_artist, en_name, en_album, en_artist
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
            a, b, c = ko.sub(' ', name).strip(), ko.sub(' ', album).strip(), ko.sub(' ', artist).strip()
            d, e, f = en.sub(' ', name).strip(), en.sub(' ', album).strip(), en.sub(' ', artist).strip()
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


def ko_extract_tag_from(data, api, tag="NNG", verbose=True, limit=None):
    '''
    parameters \\
        data : dictionary \\
        api : khaiii.KhaiiiApi (tokenizer) \\
        tag : str or list of str \\
        verbose : True or False \\
        limit: int or None(default); iteration limit \\
    returns \\
        dictionary
    '''
    extracted = {}
    if limit:
        _items = tqdm(list(data.items())[:limit]) if verbose else list(data.items())[:limit]
    else:
        _items = tqdm(data.items()) if verbose else data.items()
    for key, val in _items:
        tokens = []
        for word in api.analyze(val):
            for morph in word.morphs:
                if morph.tag in tag:
                    tokens.append(morph.lex)
        if tokens != []:
            extracted[key] = tokens
    return extracted

def en_extract_tag_from(data, api, tag, verbose=True, limit=None):
    pass

def flatten_dict():
    pass

def we_numbering(data, start=0, verbose=True, limit=None, return_inverse=False):
    '''
    data : list of dictionaries
    start : int; start numbering index. default is 0.
    return_inverse : bool; return dictionary idx -> word
    '''
    idx = start
    words = {}
    if return_inverse:
        inverse = {}
    for i, _dict in enumerate(data):
        print(f"{i+1}/{len(data)}") if verbose else None
        if limit:
            _items = list(_dict.items())[:limit]
        else:
            _items = tqdm(_dict.items()) if verbose else _dict.items()
        for _, vals in _items:
            for val in vals:
                if not(val in words):
                    words[val] = idx
                    if return_inverse:
                        inverse[idx] = val
                    idx += 1
    if return_inverse:
        return words, inverse
    else:
        return words


def n_gram(n, data):
    dct = {}
    for key, title in data.items():
        bag = []
        for pos in range(len(title) - n + 1):
            bag.append(title[pos: pos + n])
        dct[key] = bag
    return dct


def n_gram_sentence(n, sentence):
    bag = []
    for pos in range(len(sentence) - n + 1):
        bag.append(sentence[pos: pos + n])
    return bag


if __name__=="__main__":
    # train = pd.read_json("res/train.json")[["id", "plylst_title", "songs", "tags", "like_cnt", "updt_date"]]
    # a, b, c, d = extract_playlist_from(train, dtype="str", dense=True, limit=100)
    # print(a, b, c, d)
    # song = pd.read_json("res/song_meta.json")[["id", "song_name", "artist_id_basket", "artist_name_basket",\
    #                                         "album_id", "album_name", "song_gn_gnr_basket",\
    #                                         "song_gn_dtl_gnr_basket", "issue_date"]]
    # a, b, c, d, e, f = extract_song_from(song, limit=100)
    # print(a, b, c, d, e, f)
    data = [{1:[10,11], 2:[20, 22], 3:[30, 33], -1:[-10, -12]}]
    words = we_numbering(data, limit=10)
    print(words)