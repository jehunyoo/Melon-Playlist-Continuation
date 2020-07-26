import pandas as pd
from arena_util import most_popular, remove_seen


def tag_id_meta(train, val):
    '''
    train, val : list of pandas.DataFrame
    @returns : (dictionary, dictionary)
    '''

    tag_to_id = {}
    id_to_tag = {}
    data = [train, val]

    tag_id = 0
    for df in data:
        for idx in df.index:
            for tag in df["tags"][idx]:
                if tag not in tag_to_id:
                    tag_to_id[tag] = tag_id
                    id_to_tag[tag_id] = tag
                    tag_id += 1
    return tag_to_id, id_to_tag


def convert_tag_to_id(data, tag_to_id):
    '''
    data : pandas.DataFrame
    tag_to_id : dictionary
    @returns : pandas.DataFrame
    '''

    data = data.copy()
    for idx in data.index:
        new_tags = []
        for tag in data["tags"][idx]:
            new_tags.append(tag_to_id[tag])
        data.at[idx, "tags"] = new_tags
    return data


def convert_id_to_tag(data, id_to_tag):
    '''
    data : pandas.DataFrame
    id_to_tag : dictionary
    @returns : pandas.DataFrame
    '''

    data = data.copy()
    for idx in data.index:
        new_tags = []
        for tag_id in data["tags"][idx]:
            new_tags.append(id_to_tag[tag_id])
        data.at[idx, "tags"] = new_tags
    return data


def to_list(df):
    '''
    df : pandas.DataFrame
    @returns : list
    '''

    lst = []
    for idx in df.index:
        dct = dict()
        dct["id"] = df["id"][idx]
        dct["songs"] = df["songs"][idx]
        dct["tags"] = df["tags"][idx]
        lst.append(dct)
    return lst


def generate_answers(train, questions):

    _, song_mp = most_popular(train, "songs", 200)
    _, tag_mp = most_popular(train, "tags", 100)

    answers = []

    for q in questions:
        if len(q["songs"]) != 0 and len(q["tags"]) != 0:
            answers.append({
                "id": q["id"],
                "songs": q["songs"],
                "tags": q["tags"]
            })
        else:
            answers.append({
                "id": q["id"],
                "songs": remove_seen(q["songs"], song_mp)[:100],
                "tags": remove_seen(q["tags"], tag_mp)[:10]
            })

    return answers


if __name__ == "__main__":
    pass
